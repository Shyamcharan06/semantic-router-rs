use anyhow::{anyhow, Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use tokenizers::Tokenizer;

/// Wraps a Candle BERT/MiniLM model and produces mean-pooled, L2-normalized
/// sentence embeddings entirely in-process (no Python, no ONNX runtime).
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub async fn load(model_id: &str, revision: &str) -> Result<Self> {
        let (owner, name) = model_id
            .split_once('/')
            .ok_or_else(|| anyhow!("embedding.model_id must be in 'owner/name' form, got '{model_id}'"))?;

        let client = hf_hub::HFClient::new().context("failed to create Hugging Face Hub client")?;
        let repo = client.model(owner, name);

        // Download only the three files we actually need. `snapshot_download`
        // would also pull the PyTorch/TF/ONNX/OpenVINO variants that ship
        // alongside the safetensors weights in this repo (several hundred
        // extra MB), which made downloads hang well past any reasonable
        // timeout.
        let config_path = repo
            .download_file()
            .filename("config.json")
            .revision(revision)
            .send()
            .await
            .with_context(|| format!("failed to download config.json for {model_id}@{revision}"))?;
        let tokenizer_path = repo
            .download_file()
            .filename("tokenizer.json")
            .revision(revision)
            .send()
            .await
            .with_context(|| format!("failed to download tokenizer.json for {model_id}@{revision}"))?;
        let weights_path = repo
            .download_file()
            .filename("model.safetensors")
            .revision(revision)
            .send()
            .await
            .with_context(|| format!("failed to download model.safetensors for {model_id}@{revision}"))?;

        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {config_path:?}"))?;
        let config: BertConfig = serde_json::from_str(&config_str).context("failed to parse BERT config.json")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("failed to load tokenizer from {tokenizer_path:?}: {e}"))?;

        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DTYPE, &device)
                .with_context(|| format!("failed to load model weights from {weights_path:?}"))?
        };
        let model = BertModel::load(vb, &config).context("failed to construct BertModel")?;

        Ok(Self { model, tokenizer, device })
    }

    /// Embeds a single piece of text into a mean-pooled, L2-normalized vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenization failed: {e}"))?;
        let ids = encoding.get_ids();
        let token_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let output = self.model.forward(&token_ids, &token_type_ids, None)?;

        let (_batch, n_tokens, _hidden) = output.dims3()?;
        let pooled = (output.sum(1)? / (n_tokens as f64))?;
        let normalized = normalize_l2(&pooled)?;
        let vec: Vec<f32> = normalized.squeeze(0)?.to_vec1()?;
        Ok(vec)
    }
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
