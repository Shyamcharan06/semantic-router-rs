use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct ClassifierWeights {
    labels: Vec<String>,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

/// A linear probe (logistic regression: one row of weights + a bias per
/// category, softmax over the logits) trained on frozen MiniLM embeddings --
/// see `eval/train_classifier.py`. An alternative to pure cosine-similarity
/// routing, selected via `routing.strategy: classifier` in `routes.yaml`.
pub struct Classifier {
    labels: Vec<String>,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

impl Classifier {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("failed to read classifier weights file {:?}", path.as_ref()))?;
        let parsed: ClassifierWeights =
            serde_json::from_str(&raw).context("failed to parse classifier weights JSON")?;

        if parsed.weights.len() != parsed.labels.len() || parsed.bias.len() != parsed.labels.len() {
            bail!("classifier weights file has mismatched labels/weights/bias lengths");
        }
        if let Some(row) = parsed.weights.first() {
            if parsed.weights.iter().any(|w| w.len() != row.len()) {
                bail!("classifier weights file has inconsistent embedding dimensions across labels");
            }
        }

        Ok(Self { labels: parsed.labels, weights: parsed.weights, bias: parsed.bias })
    }

    /// Returns the predicted label and its softmax probability.
    pub fn predict(&self, embedding: &[f32]) -> (String, f32) {
        let logits: Vec<f32> =
            self.weights.iter().zip(&self.bias).map(|(w, b)| dot(w, embedding) + b).collect();

        let max_logit = logits.iter().cloned().fold(f32::MIN, f32::max);
        let exp: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = exp.iter().sum();

        let (best_idx, best_logit_exp) =
            exp.iter().enumerate().fold((0usize, f32::MIN), |acc, (i, &e)| if e > acc.1 { (i, e) } else { acc });

        (self.labels[best_idx].clone(), best_logit_exp / sum)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_classifier() -> Classifier {
        // 2 labels, 2-dim toy embeddings: "a" fires on [1, 0], "b" on [0, 1].
        Classifier {
            labels: vec!["a".to_string(), "b".to_string()],
            weights: vec![vec![5.0, 0.0], vec![0.0, 5.0]],
            bias: vec![0.0, 0.0],
        }
    }

    #[test]
    fn predicts_highest_scoring_label() {
        let clf = toy_classifier();
        let (label, prob) = clf.predict(&[1.0, 0.0]);
        assert_eq!(label, "a");
        assert!(prob > 0.9, "expected high confidence, got {prob}");
    }

    #[test]
    fn predicts_the_other_label() {
        let clf = toy_classifier();
        let (label, _) = clf.predict(&[0.0, 1.0]);
        assert_eq!(label, "b");
    }

    #[test]
    fn ambiguous_input_gives_lower_confidence() {
        let clf = toy_classifier();
        let (_, prob) = clf.predict(&[0.5, 0.5]);
        assert!(prob < 0.9, "expected lower confidence for ambiguous input, got {prob}");
    }

    #[test]
    fn rejects_mismatched_label_and_weight_counts() {
        let json = r#"{"labels": ["a", "b"], "weights": [[1.0, 2.0]], "bias": [0.0, 0.0]}"#;
        let dir = std::env::temp_dir().join("semantic_router_classifier_test_mismatch.json");
        std::fs::write(&dir, json).unwrap();
        let result = Classifier::load(&dir);
        std::fs::remove_file(&dir).ok();
        assert!(result.is_err());
    }
}
