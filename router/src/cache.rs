use crate::routing::cosine_similarity;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

#[derive(Clone)]
struct CacheEntry {
    embedding: Vec<f32>,
    response: serde_json::Value,
    category: Option<String>,
    inserted_at: Instant,
}

pub struct CacheHit {
    pub response: serde_json::Value,
    pub category: Option<String>,
    pub score: f32,
}

/// A brute-force semantic cache: near-duplicate prompts (by cosine
/// similarity) return a cached response instead of hitting the backend LLM.
/// Fine at the scale of a single router instance; would need a real vector
/// index if the entry count grew into the tens of thousands.
pub struct SemanticCache {
    entries: Mutex<VecDeque<CacheEntry>>,
    similarity_threshold: f32,
    ttl: Duration,
    max_entries: usize,
    enabled: bool,
}

impl SemanticCache {
    pub fn new(
        enabled: bool,
        similarity_threshold: f32,
        ttl_seconds: u64,
        max_entries: usize,
    ) -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
            similarity_threshold,
            ttl: Duration::from_secs(ttl_seconds),
            max_entries,
            enabled,
        }
    }

    pub async fn get(&self, query_embedding: &[f32]) -> Option<CacheHit> {
        if !self.enabled {
            return None;
        }

        let mut entries = self.entries.lock().await;
        let now = Instant::now();
        entries.retain(|e| now.duration_since(e.inserted_at) < self.ttl);

        let mut best: Option<(usize, f32)> = None;
        for (i, e) in entries.iter().enumerate() {
            let score = cosine_similarity(query_embedding, &e.embedding);
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((i, score));
            }
        }

        match best {
            Some((i, score)) if score >= self.similarity_threshold => {
                let entry = &entries[i];
                Some(CacheHit {
                    response: entry.response.clone(),
                    category: entry.category.clone(),
                    score,
                })
            }
            _ => None,
        }
    }

    pub async fn put(
        &self,
        embedding: Vec<f32>,
        response: serde_json::Value,
        category: Option<String>,
    ) {
        if !self.enabled {
            return;
        }

        let mut entries = self.entries.lock().await;
        if entries.len() >= self.max_entries {
            entries.pop_front();
        }
        entries.push_back(CacheEntry {
            embedding,
            response,
            category,
            inserted_at: Instant::now(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn returns_hit_for_near_duplicate_embedding() {
        let cache = SemanticCache::new(true, 0.9, 300, 10);
        cache
            .put(
                vec![1.0, 0.0],
                json!({"answer": "cached"}),
                Some("coding".into()),
            )
            .await;

        let hit = cache.get(&[0.99, 0.01]).await;
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().response, json!({"answer": "cached"}));
    }

    #[tokio::test]
    async fn returns_none_below_similarity_threshold() {
        let cache = SemanticCache::new(true, 0.99, 300, 10);
        cache
            .put(vec![1.0, 0.0], json!({"answer": "cached"}), None)
            .await;

        let hit = cache.get(&[0.0, 1.0]).await;
        assert!(hit.is_none());
    }

    #[tokio::test]
    async fn disabled_cache_never_hits() {
        let cache = SemanticCache::new(false, 0.5, 300, 10);
        cache
            .put(vec![1.0, 0.0], json!({"answer": "cached"}), None)
            .await;

        let hit = cache.get(&[1.0, 0.0]).await;
        assert!(hit.is_none());
    }

    #[tokio::test]
    async fn evicts_oldest_entry_when_full() {
        let cache = SemanticCache::new(true, 0.99, 300, 1);
        cache
            .put(vec![1.0, 0.0], json!({"answer": "first"}), None)
            .await;
        cache
            .put(vec![0.0, 1.0], json!({"answer": "second"}), None)
            .await;

        assert!(cache.get(&[1.0, 0.0]).await.is_none());
        assert!(cache.get(&[0.0, 1.0]).await.is_some());
    }
}
