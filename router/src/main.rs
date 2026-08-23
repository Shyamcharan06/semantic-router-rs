use anyhow::Result;
use semantic_router::{cache, config, embeddings, proxy, routing, server, telemetry};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Kept alive for the process lifetime: dropping it would shut down the
    // batch span processor's background export task.
    let _tracer_provider = telemetry::init();

    let config_path =
        std::env::var("ROUTER_CONFIG").unwrap_or_else(|_| "config/routes.yaml".to_string());
    let cfg = config::Config::load(&config_path)?;

    tracing::info!(model = %cfg.embedding.model_id, revision = %cfg.embedding.revision, "loading embedding model");
    let embedder =
        embeddings::Embedder::load(&cfg.embedding.model_id, &cfg.embedding.revision).await?;

    let mut category_indexes = Vec::with_capacity(cfg.categories.len());
    let mut backends = HashMap::with_capacity(cfg.categories.len());

    for cat in &cfg.categories {
        let mut embs = Vec::with_capacity(cat.examples.len());
        for example in &cat.examples {
            embs.push(embedder.embed(example)?);
        }
        tracing::info!(category = %cat.name, examples = cat.examples.len(), "indexed category");

        category_indexes.push(routing::CategoryIndex {
            name: cat.name.clone(),
            embeddings: embs,
        });
        backends.insert(
            cat.name.clone(),
            proxy::BackendTarget {
                base_url: cat.backend.base_url.clone(),
                api_key: cat
                    .backend
                    .api_key_env
                    .as_ref()
                    .and_then(|k| std::env::var(k).ok()),
                model: cat.backend.model.clone(),
            },
        );
    }

    let similarity_router =
        routing::SemanticRouter::new(category_indexes, cfg.routing.confidence_threshold);

    let routing_strategy = match cfg.routing.strategy {
        config::RoutingStrategyKind::Similarity => {
            routing::RoutingStrategy::Similarity(similarity_router)
        }
        config::RoutingStrategyKind::Classifier => {
            let classifier =
                semantic_router::classifier::Classifier::load(&cfg.routing.classifier_path)?;
            tracing::info!(path = %cfg.routing.classifier_path, "loaded trained routing classifier");
            routing::RoutingStrategy::Classifier {
                classifier,
                confidence_threshold: cfg.routing.confidence_threshold,
            }
        }
    };

    let default_backend = proxy::BackendTarget {
        base_url: cfg.default.base_url.clone(),
        api_key: cfg
            .default
            .api_key_env
            .as_ref()
            .and_then(|k| std::env::var(k).ok()),
        model: cfg.default.model.clone(),
    };

    let cache = cache::SemanticCache::new(
        cfg.cache.enabled,
        cfg.cache.similarity_threshold,
        cfg.cache.ttl_seconds,
        cfg.cache.max_entries,
    );

    let prompt_guard =
        semantic_router::prompt_guard::PromptGuard::new(&cfg.security.prompt_guard.extra_patterns);

    let state = Arc::new(server::AppState {
        embedder: Arc::new(embedder),
        router: Arc::new(routing_strategy),
        backends,
        default_backend,
        default_backend_name: cfg.default.name.clone(),
        proxy: proxy::Proxy::new(),
        cache: Arc::new(cache),
        security: cfg.security.clone(),
        prompt_guard,
    });

    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let app = server::build_router(state);

    tracing::info!(%addr, "semantic router listening");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
