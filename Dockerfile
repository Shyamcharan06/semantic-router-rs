# Builds the Rust router. Candle runs pure-Rust CPU inference, so no CUDA/
# ONNX runtime image is needed -- just a C toolchain for a couple of native
# dependencies (zstd, the tokenizer backend), which the official rust image
# already ships.
FROM rust:1-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY router ./router
RUN cargo build --release -p semantic-router

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/semantic-router /usr/local/bin/semantic-router
COPY config ./config
ENV ROUTER_CONFIG=/app/config/routes.yaml
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/semantic-router"]
