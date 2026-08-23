use crate::cache::SemanticCache;
use crate::config::{PiiAction, SecurityConfig};
use crate::embeddings::Embedder;
use crate::pii;
use crate::prompt_guard::PromptGuard;
use crate::proxy::{BackendTarget, Proxy};
use crate::routing::RoutingStrategy;
use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing::Instrument;

pub struct AppState {
    pub embedder: Arc<Embedder>,
    pub router: Arc<RoutingStrategy>,
    pub backends: HashMap<String, BackendTarget>,
    pub default_backend: BackendTarget,
    pub default_backend_name: String,
    pub proxy: Proxy,
    pub cache: Arc<SemanticCache>,
    pub security: SecurityConfig,
    pub prompt_guard: PromptGuard,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/route", get(route_debug))
        .route("/embed", get(embed_debug))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": { "type": "internal_error", "message": self.0.to_string() } });
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
    let model = target
        .map(|t| t.model.clone())
        .unwrap_or_else(|| state.default_backend.model.clone());

    Ok(Json(serde_json::json!({
        "query": params.q,
        "category": decision.category,
        "score": decision.score,
        "model": model,
    })))
}

/// Raw embedding for a piece of text -- used by eval/train_classifier.py so
/// there's exactly one embedding implementation (this one) that both
/// routing strategies are trained/evaluated against, never a second copy
/// re-implemented in Python.
async fn embed_debug(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RouteQuery>,
) -> Result<Json<Value>, AppError> {
    let embedding = state.embedder.embed(&params.q)?;
    Ok(Json(serde_json::json!({ "embedding": embedding })))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(mut body): Json<Value>,
) -> Result<Response, AppError> {
    let mut query_text = extract_last_user_message(&body).ok_or_else(|| {
        anyhow::anyhow!("request must include a message with role 'user' and string content")
    })?;

    if state.security.prompt_guard.enabled {
        let matched = tracing::info_span!("prompt_guard").in_scope(|| {
            state
                .prompt_guard
                .matched_pattern(&query_text)
                .map(str::to_string)
        });
        if let Some(pattern) = matched {
            return Ok(security_block_response(
                StatusCode::FORBIDDEN,
                "prompt_guard_triggered",
                format!("request blocked: matched a prompt-guard pattern ('{pattern}')"),
            ));
        }
    }

    if state.security.pii.enabled {
        let findings = tracing::info_span!("pii_scan").in_scope(|| pii::scan(&query_text));
        if !findings.is_empty() {
            match state.security.pii.action {
                PiiAction::Block => {
                    let kinds: Vec<&str> = findings.iter().map(|f| f.kind).collect();
                    return Ok(security_block_response(
                        StatusCode::BAD_REQUEST,
                        "pii_detected",
                        format!("request blocked: detected PII ({})", kinds.join(", ")),
                    ));
                }
                PiiAction::Redact => {
                    let (redacted_text, _found) =
                        tracing::info_span!("pii_redact").in_scope(|| pii::redact(&query_text));
                    set_last_user_message(&mut body, &redacted_text);
                    query_text = redacted_text;
                }
            }
        }
    }

    let embedding = tracing::info_span!("embed").in_scope(|| state.embedder.embed(&query_text))?;
    let is_streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    if !is_streaming {
        let cache_hit = state
            .cache
            .get(&embedding)
            .instrument(tracing::info_span!("cache_lookup"))
            .await;
        if let Some(hit) = cache_hit {
            let category_label = hit
                .category
                .clone()
                .unwrap_or_else(|| state.default_backend_name.clone());
            let headers = routing_headers(&category_label, hit.score, "hit");
            return Ok((headers, Json(hit.response)).into_response());
        }
    }

    let decision = tracing::info_span!("route").in_scope(|| state.router.route(&embedding));
    let target = match &decision.category {
        Some(name) => state
            .backends
            .get(name)
            .cloned()
            .unwrap_or_else(|| state.default_backend.clone()),
        None => state.default_backend.clone(),
    };
    let category_label = decision
        .category
        .clone()
        .unwrap_or_else(|| state.default_backend_name.clone());

    if is_streaming {
        let upstream = state
            .proxy
            .forward_chat_completion_stream(&target, body)
            .instrument(tracing::info_span!("proxy_backend_call", backend = %target.base_url, model = %target.model, streaming = true))
            .await?;
        // Pass through whatever content-type the backend actually sent
        // (a backend that doesn't support streaming might just return
        // ordinary JSON even for a `stream: true` request) rather than
        // assuming SSE.
        let content_type = upstream
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/event-stream")
            .to_string();
        let byte_stream = upstream.bytes_stream();

        let mut response = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", content_type)
            .body(Body::from_stream(byte_stream))
            .map_err(|e| anyhow::anyhow!("failed to build streaming response: {e}"))?;
        apply_routing_headers(
            response.headers_mut(),
            &category_label,
            decision.score,
            "bypassed",
        );
        return Ok(response);
    }

    let response = state
        .proxy
        .forward_chat_completion(&target, body)
        .instrument(tracing::info_span!("proxy_backend_call", backend = %target.base_url, model = %target.model, streaming = false))
        .await?;
    state
        .cache
        .put(embedding, response.clone(), decision.category.clone())
        .instrument(tracing::info_span!("cache_put"))
        .await;

    let headers = routing_headers(&category_label, decision.score, "miss");
    Ok((headers, Json(response)).into_response())
}

fn security_block_response(status: StatusCode, error_type: &str, message: String) -> Response {
    let body = serde_json::json!({ "error": { "type": error_type, "message": message } });
    (status, Json(body)).into_response()
}

fn routing_headers(category: &str, score: f32, cache_status: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    apply_routing_headers(&mut headers, category, score, cache_status);
    headers
}

fn apply_routing_headers(headers: &mut HeaderMap, category: &str, score: f32, cache_status: &str) {
    headers.insert("x-router-category", header_value(category));
    headers.insert("x-router-score", header_value(&score.to_string()));
    headers.insert("x-router-cache", header_value(cache_status));
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

/// Overwrites the content of the last `role: user` message in-place, e.g.
/// after PII redaction, so the (possibly rewritten) request is what
/// actually gets forwarded and embedded.
fn set_last_user_message(body: &mut Value, new_text: &str) -> bool {
    let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return false;
    };
    for message in messages.iter_mut().rev() {
        if message.get("role").and_then(|r| r.as_str()) == Some("user")
            && let Some(content) = message.get_mut("content")
        {
            *content = Value::String(new_text.to_string());
            return true;
        }
    }
    false
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
        assert_eq!(
            extract_last_user_message(&body).as_deref(),
            Some("second question")
        );
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

    #[test]
    fn set_last_user_message_overwrites_content() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ]
        });
        assert!(set_last_user_message(&mut body, "redacted"));
        assert_eq!(body["messages"][1]["content"], "redacted");
        assert_eq!(body["messages"][0]["content"], "first");
    }

    #[test]
    fn set_last_user_message_returns_false_when_no_user_message() {
        let mut body = json!({"messages": [{"role": "system", "content": "be nice"}]});
        assert!(!set_last_user_message(&mut body, "redacted"));
    }
}
