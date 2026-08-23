use crate::cache::SemanticCache;
use crate::embeddings::Embedder;
use crate::proxy::{BackendTarget, Proxy};
use crate::routing::SemanticRouter;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub struct AppState {
    pub embedder: Arc<Embedder>,
    pub router: Arc<SemanticRouter>,
    pub backends: HashMap<String, BackendTarget>,
    pub default_backend: BackendTarget,
    pub default_backend_name: String,
    pub proxy: Proxy,
    pub cache: Arc<SemanticCache>,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/route", get(route_debug))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": self.0.to_string() });
        (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

async fn healthz() -> &'static str {
    "ok"
}

#[derive(Deserialize)]
struct RouteQuery {
    q: String,
}

async fn route_debug(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RouteQuery>,
) -> Result<Json<Value>, AppError> {
    let embedding = state.embedder.embed(&params.q)?;
    let decision = state.router.route(&embedding);

    let target = match &decision.category {
        Some(name) => state.backends.get(name),
        None => None,
    };
    let model = target.map(|t| t.model.clone()).unwrap_or_else(|| state.default_backend.model.clone());

    Ok(Json(serde_json::json!({
        "query": params.q,
        "category": decision.category,
        "score": decision.score,
        "model": model,
    })))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<impl IntoResponse, AppError> {
    let query_text = extract_last_user_message(&body)
        .ok_or_else(|| anyhow::anyhow!("request must include a message with role 'user' and string content"))?;

    let embedding = state.embedder.embed(&query_text)?;

    if let Some(hit) = state.cache.get(&embedding).await {
        let category_label = hit.category.clone().unwrap_or_else(|| state.default_backend_name.clone());
        let headers = routing_headers(&category_label, hit.score, "hit");
        return Ok((headers, Json(hit.response)));
    }

    let decision = state.router.route(&embedding);
    let target = match &decision.category {
        Some(name) => state.backends.get(name).cloned().unwrap_or_else(|| state.default_backend.clone()),
        None => state.default_backend.clone(),
    };

    let response = state.proxy.forward_chat_completion(&target, body).await?;
    state.cache.put(embedding, response.clone(), decision.category.clone()).await;

    let category_label = decision.category.clone().unwrap_or_else(|| state.default_backend_name.clone());
    let headers = routing_headers(&category_label, decision.score, "miss");

    Ok((headers, Json(response)))
}

fn routing_headers(category: &str, score: f32, cache_status: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("x-router-category", header_value(category));
    headers.insert("x-router-score", header_value(&score.to_string()));
    headers.insert("x-router-cache", header_value(cache_status));
    headers
}

fn header_value(s: &str) -> HeaderValue {
    HeaderValue::from_str(s).unwrap_or_else(|_| HeaderValue::from_static("invalid"))
}

fn extract_last_user_message(body: &Value) -> Option<String> {
    body.get("messages")?
        .as_array()?
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_last_user_message() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "an answer"},
                {"role": "user", "content": "second question"},
            ]
        });
        assert_eq!(extract_last_user_message(&body).as_deref(), Some("second question"));
    }

    #[test]
    fn returns_none_when_no_user_message() {
        let body = json!({"messages": [{"role": "system", "content": "be nice"}]});
        assert_eq!(extract_last_user_message(&body), None);
    }

    #[test]
    fn returns_none_when_messages_missing() {
        let body = json!({});
        assert_eq!(extract_last_user_message(&body), None);
    }
}
