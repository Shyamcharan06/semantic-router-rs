use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self { host: default_host(), port: default_port() }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    pub model_id: String,
    #[serde(default = "default_revision")]
    pub revision: String,
}

fn default_revision() -> String {
    "main".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct RoutingConfig {
    #[serde(default = "default_threshold")]
    pub confidence_threshold: f32,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self { confidence_threshold: default_threshold() }
    }
}

fn default_threshold() -> f32 {
    0.35
}

#[derive(Debug, Clone, Deserialize)]
pub struct CacheConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_cache_threshold")]
    pub similarity_threshold: f32,
    #[serde(default = "default_ttl")]
    pub ttl_seconds: u64,
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            similarity_threshold: default_cache_threshold(),
            ttl_seconds: default_ttl(),
            max_entries: default_max_entries(),
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_cache_threshold() -> f32 {
    0.92
}

fn default_ttl() -> u64 {
    300
}

fn default_max_entries() -> usize {
    1000
}

#[derive(Debug, Clone, Deserialize)]
pub struct Backend {
    pub base_url: String,
    pub api_key_env: Option<String>,
    pub model: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Category {
    pub name: String,
    pub examples: Vec<String>,
    pub backend: Backend,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DefaultRoute {
    pub name: String,
    pub base_url: String,
    pub api_key_env: Option<String>,
    pub model: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub routing: RoutingConfig,
    pub default: DefaultRoute,
    #[serde(default)]
    pub cache: CacheConfig,
    pub categories: Vec<Category>,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("failed to read config file {:?}", path.as_ref()))?;
        let cfg: Config = serde_yaml::from_str(&raw).context("failed to parse config as YAML")?;
        if cfg.categories.is_empty() {
            bail!("config must define at least one category");
        }
        for cat in &cfg.categories {
            if cat.examples.is_empty() {
                bail!("category '{}' must have at least one example utterance", cat.name);
            }
        }
        Ok(cfg)
    }
}
