//! End-to-end test: a real (in-process, CPU) Candle/MiniLM embedder routes
//! prompts to the right category, which gets proxied to a mock backend that
//! just echoes back which model it received. No real LLM API key needed.
//!
//! The first run downloads ~90MB of model weights from the Hugging Face Hub
//! into the local HF cache; subsequent runs reuse the cache and are fast.

use axum::routing::post;
use axum::{Json, Router};
use semantic_router::cache::SemanticCache;
use semantic_router::embeddings::Embedder;
use semantic_router::proxy::{BackendTarget, Proxy};
use semantic_router::routing::{CategoryIndex, SemanticRouter};
use semantic_router::server::{build_router, AppState};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::OnceCell;

/// Both integration tests need the same MiniLM model. Loading it twice in
/// parallel makes them race on the Hugging Face Hub cache's download lock,
/// so the two tests in this binary share a single instance instead.
static EMBEDDER: OnceCell<Arc<Embedder>> = OnceCell::const_new();

async fn shared_embedder() -> Arc<Embedder> {
    EMBEDDER
        .get_or_init(|| async {
            Arc::new(
                Embedder::load("sentence-transformers/all-MiniLM-L6-v2", "main")
                    .await
                    .expect("failed to load embedding model - requires network access on first run"),
            )
        })
        .await
        .clone()
}

async fn echo_model(Json(body): Json<Value>) -> Json<Value> {
    let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("unknown");
    Json(json!({
        "echo_model": model,
        "choices": [{"message": {"role": "assistant", "content": "mock response"}}]
    }))
}

async fn spawn_mock_backend() -> SocketAddr {
    let app = Router::new().route("/v1/chat/completions", post(echo_model));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

async fn spawn_router_server(backend_addr: SocketAddr) -> SocketAddr {
    let embedder = shared_embedder().await;

    let categories = [
        ("coding", vec![
            "Write a Python function to reverse a linked list",
            "Why is my Rust borrow checker complaining about this code?",
            "Debug this JavaScript async function",
            "Explain the time complexity of quicksort",
        ]),
        ("creative_writing", vec![
            "Write a short story about a lighthouse keeper",
            "Compose a poem about autumn leaves",
            "Give me a plot twist for a mystery novel",
            "Write dialogue between two old friends",
        ]),
    ];

    let mut category_indexes = Vec::new();
    let mut backends = HashMap::new();
    for (name, examples) in categories {
        let embeddings = examples.iter().map(|e| embedder.embed(e).unwrap()).collect();
        category_indexes.push(CategoryIndex { name: name.to_string(), embeddings });
        backends.insert(
            name.to_string(),
            BackendTarget {
                base_url: format!("http://{backend_addr}"),
                api_key: None,
                model: format!("{name}-model"),
            },
        );
    }

    let router = SemanticRouter::new(category_indexes, 0.35);
    let default_backend = BackendTarget {
        base_url: format!("http://{backend_addr}"),
        api_key: None,
        model: "general-model".to_string(),
    };

    let state = Arc::new(AppState {
        embedder,
        router: Arc::new(router),
        backends,
        default_backend,
        default_backend_name: "default".to_string(),
        proxy: Proxy::new(),
        cache: Arc::new(SemanticCache::new(true, 0.92, 300, 1000)),
    });

    let app = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

#[tokio::test]
async fn routes_coding_and_creative_prompts_to_the_right_backend() {
    let backend_addr = spawn_mock_backend().await;
    let router_addr = spawn_router_server(backend_addr).await;

    let client = reqwest::Client::new();

    let coding_resp = client
        .post(format!("http://{router_addr}/v1/chat/completions"))
        .json(&json!({
            "model": "unused",
            "messages": [{"role": "user", "content": "How do I fix a null pointer exception in Java?"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(coding_resp.headers().get("x-router-category").unwrap(), "coding");
    let coding_body: Value = coding_resp.json().await.unwrap();
    assert_eq!(coding_body["echo_model"], "coding-model");

    let creative_resp = client
        .post(format!("http://{router_addr}/v1/chat/completions"))
        .json(&json!({
            "model": "unused",
            "messages": [{"role": "user", "content": "Write me a poem about a rainy city night"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(creative_resp.headers().get("x-router-category").unwrap(), "creative_writing");
    let creative_body: Value = creative_resp.json().await.unwrap();
    assert_eq!(creative_body["echo_model"], "creative_writing-model");
}

#[tokio::test]
async fn route_debug_endpoint_returns_decision_without_calling_backend() {
    let backend_addr = spawn_mock_backend().await;
    let router_addr = spawn_router_server(backend_addr).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("http://{router_addr}/route"))
        .query(&[("q", "Debug this segfault in my C++ program")])
        .send()
        .await
        .unwrap();

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["category"], "coding");
    assert_eq!(body["model"], "coding-model");
}
