use anyhow::{Result, anyhow};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct BackendTarget {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
}

/// Forwards OpenAI-style chat completion requests to whichever backend the
/// router selected, rewriting the `model` field to that backend's model name.
pub struct Proxy {
    client: reqwest::Client,
}

impl Default for Proxy {
    fn default() -> Self {
        Self::new()
    }
}

impl Proxy {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    pub async fn forward_chat_completion(
        &self,
        target: &BackendTarget,
        mut body: Value,
    ) -> Result<Value> {
        if let Some(obj) = body.as_object_mut() {
            obj.insert("model".to_string(), Value::String(target.model.clone()));
        }

        let url = format!(
            "{}/v1/chat/completions",
            target.base_url.trim_end_matches('/')
        );
        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &target.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| anyhow!("request to backend {url} failed: {e}"))?;
        let status = resp.status();
        let payload: Value = resp
            .json()
            .await
            .map_err(|e| anyhow!("failed to parse response from backend {url}: {e}"))?;

        if !status.is_success() {
            return Err(anyhow!("backend {url} returned {status}: {payload}"));
        }

        Ok(payload)
    }

    /// Forwards a `"stream": true` chat completion and returns the raw
    /// backend response so its body can be piped straight through as
    /// Server-Sent Events, without buffering the whole thing in memory.
    pub async fn forward_chat_completion_stream(
        &self,
        target: &BackendTarget,
        mut body: Value,
    ) -> Result<reqwest::Response> {
        if let Some(obj) = body.as_object_mut() {
            obj.insert("model".to_string(), Value::String(target.model.clone()));
        }

        let url = format!(
            "{}/v1/chat/completions",
            target.base_url.trim_end_matches('/')
        );
        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &target.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| anyhow!("request to backend {url} failed: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let payload = resp.text().await.unwrap_or_default();
            return Err(anyhow!("backend {url} returned {status}: {payload}"));
        }

        Ok(resp)
    }
}
