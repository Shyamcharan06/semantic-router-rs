use crate::classifier::Classifier;

/// A category with its example utterances already embedded at startup.
#[derive(Debug, Clone)]
pub struct CategoryIndex {
    pub name: String,
    pub embeddings: Vec<Vec<f32>>,
}

/// The two ways an embedding gets turned into a routing decision: pure
/// max-cosine-similarity against example utterances (no training), or a
/// small trained linear classifier (see eval/train_classifier.py). Picked
/// via `routing.strategy` in `routes.yaml`.
pub enum RoutingStrategy {
    Similarity(SemanticRouter),
    Classifier { classifier: Classifier, confidence_threshold: f32 },
}

impl RoutingStrategy {
    pub fn route(&self, query_embedding: &[f32]) -> RouteDecision {
        match self {
            RoutingStrategy::Similarity(router) => router.route(query_embedding),
            RoutingStrategy::Classifier { classifier, confidence_threshold } => {
                let (label, confidence) = classifier.predict(query_embedding);
                if confidence >= *confidence_threshold {
                    RouteDecision { category: Some(label), score: confidence }
                } else {
                    RouteDecision { category: None, score: confidence }
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouteDecision {
    /// `None` means no category cleared the confidence threshold; callers
    /// should fall back to the configured default backend.
    pub category: Option<String>,
    pub score: f32,
}

/// Routes a query embedding to the best-matching category by max cosine
/// similarity against that category's example embeddings.
pub struct SemanticRouter {
    categories: Vec<CategoryIndex>,
    confidence_threshold: f32,
}

impl SemanticRouter {
    pub fn new(categories: Vec<CategoryIndex>, confidence_threshold: f32) -> Self {
        Self { categories, confidence_threshold }
    }

    pub fn route(&self, query_embedding: &[f32]) -> RouteDecision {
        let mut best: Option<(&str, f32)> = None;

        for cat in &self.categories {
            let score = cat
                .embeddings
                .iter()
                .map(|e| cosine_similarity(query_embedding, e))
                .fold(f32::MIN, f32::max);

            if best.map_or(true, |(_, best_score)| score > best_score) {
                best = Some((cat.name.as_str(), score));
            }
        }

        match best {
            Some((name, score)) if score >= self.confidence_threshold => {
                RouteDecision { category: Some(name.to_string()), score }
            }
            Some((_, score)) => RouteDecision { category: None, score },
            None => RouteDecision { category: None, score: 0.0 },
        }
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_similarity_one() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors_have_similarity_negative_one() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_has_zero_similarity() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    fn make_router(threshold: f32) -> SemanticRouter {
        let categories = vec![
            CategoryIndex { name: "coding".into(), embeddings: vec![vec![1.0, 0.0, 0.0]] },
            CategoryIndex { name: "math".into(), embeddings: vec![vec![0.0, 1.0, 0.0]] },
        ];
        SemanticRouter::new(categories, threshold)
    }

    #[test]
    fn routes_to_closest_category_above_threshold() {
        let router = make_router(0.5);
        let decision = router.route(&[0.9, 0.1, 0.0]);
        assert_eq!(decision.category.as_deref(), Some("coding"));
    }

    #[test]
    fn falls_back_to_none_below_threshold() {
        let router = make_router(0.9);
        let decision = router.route(&[0.6, 0.5, 0.0]);
        assert_eq!(decision.category, None);
    }

    #[test]
    fn empty_categories_falls_back_to_none() {
        let router = SemanticRouter::new(vec![], 0.5);
        let decision = router.route(&[1.0, 0.0]);
        assert_eq!(decision.category, None);
        assert_eq!(decision.score, 0.0);
    }
}
