use opentelemetry::global;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Wires up structured console logging and, only when
/// `OTEL_EXPORTER_OTLP_ENDPOINT` is set, exports spans to an OTLP collector
/// (Jaeger in docker-compose's setup) so a single request's full pipeline --
/// prompt guard, PII scan, embedding, cache lookup, routing, backend call --
/// shows up as one trace with per-stage timing, matching the original
/// project's "fine-grained visibility into the request processing pipeline"
/// observability feature.
///
/// Returns the tracer provider so callers can flush it on shutdown; `None`
/// if OTLP export wasn't enabled/available (tracing still works via the
/// console layer either way -- this is best-effort, not load-bearing).
pub fn init() -> Option<SdkTracerProvider> {
    let env_filter = || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = || tracing_subscriber::fmt::layer();

    if std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_err() {
        tracing_subscriber::registry().with(env_filter()).with(fmt_layer()).init();
        return None;
    }

    let exporter = match opentelemetry_otlp::SpanExporter::builder().with_http().build() {
        Ok(exporter) => exporter,
        Err(e) => {
            tracing_subscriber::registry().with(env_filter()).with(fmt_layer()).init();
            tracing::warn!(error = %e, "failed to build OTLP exporter, continuing without trace export");
            return None;
        }
    };

    let provider = SdkTracerProvider::builder().with_batch_exporter(exporter).build();
    global::set_tracer_provider(provider.clone());
    let tracer = global::tracer("semantic-router");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry().with(env_filter()).with(fmt_layer()).with(otel_layer).init();
    tracing::info!("OpenTelemetry OTLP trace export enabled");
    Some(provider)
}
