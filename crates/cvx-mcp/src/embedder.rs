//! Embedding backends for inline text-to-vector conversion.
//!
//! Allows MCP tools to accept text queries instead of pre-computed vectors.
//! The `Embedder` trait is defined in cvx-core; this module provides implementations.

use cvx_core::traits::{EmbedError, Embedder};

// ─── Mock embedder (for testing) ────────────────────────────────────

/// A deterministic mock embedder that produces consistent vectors from text.
///
/// Uses a simple hash-based approach — not a real embedding model.
/// Useful for testing the MCP pipeline without loading a model.
pub struct MockEmbedder {
    dim: usize,
}

impl MockEmbedder {
    /// Create a mock embedder with the given output dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if text.is_empty() {
            return Err(EmbedError::InvalidInput("empty text".into()));
        }

        // Deterministic: hash-based pseudo-embedding
        let mut vector = vec![0.0f32; self.dim];
        for (i, byte) in text.bytes().enumerate() {
            let idx = i % self.dim;
            vector[idx] += (byte as f32 - 96.0) * 0.01;
        }

        // Normalize to unit length
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut vector {
                *v /= norm;
            }
        }

        Ok(vector)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        "mock-embedder"
    }
}

// ─── Stub for future ONNX backend ──────────────────────────────────

/// Placeholder for ONNX Runtime embedding backend.
///
/// Will be implemented when `ort` crate is added as a dependency.
/// For now, returns an error if used.
pub struct OnnxEmbedder {
    dim: usize,
    model_name: String,
}

impl OnnxEmbedder {
    /// Create a stub ONNX embedder.
    pub fn new(_model_path: &str, dim: usize) -> Self {
        Self {
            dim,
            model_name: "onnx-stub".to_string(),
        }
    }
}

impl Embedder for OnnxEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Err(EmbedError::ModelNotAvailable(
            "ONNX backend not yet implemented; enable the 'onnx' feature".into(),
        ))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

// ─── Stub for future API backend ────────────────────────────────────

/// Placeholder for remote API embedding backend (OpenAI, Cohere, etc.).
pub struct ApiEmbedder {
    dim: usize,
    model_name: String,
}

impl ApiEmbedder {
    /// Create a stub API embedder.
    pub fn new(_api_url: &str, _api_key: &str, dim: usize, model_name: &str) -> Self {
        Self {
            dim,
            model_name: model_name.to_string(),
        }
    }
}

impl Embedder for ApiEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Err(EmbedError::ModelNotAvailable(
            "API embedding backend not yet implemented".into(),
        ))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_embedder_produces_correct_dim() {
        let emb = MockEmbedder::new(384);
        let v = emb.embed("hello world").unwrap();
        assert_eq!(v.len(), 384);
    }

    #[test]
    fn mock_embedder_is_deterministic() {
        let emb = MockEmbedder::new(128);
        let v1 = emb.embed("test input").unwrap();
        let v2 = emb.embed("test input").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn mock_embedder_different_text_different_vector() {
        let emb = MockEmbedder::new(64);
        let v1 = emb.embed("hello").unwrap();
        let v2 = emb.embed("world").unwrap();
        assert_ne!(v1, v2);
    }

    #[test]
    fn mock_embedder_normalized() {
        let emb = MockEmbedder::new(128);
        let v = emb.embed("some text here").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "should be unit normalized, got {norm}"
        );
    }

    #[test]
    fn mock_embedder_empty_input() {
        let emb = MockEmbedder::new(64);
        assert!(emb.embed("").is_err());
    }

    #[test]
    fn mock_embedder_batch() {
        let emb = MockEmbedder::new(64);
        let results = emb.embed_batch(&["hello", "world"]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 64);
        assert_eq!(results[1].len(), 64);
    }

    #[test]
    fn onnx_stub_returns_error() {
        let emb = OnnxEmbedder::new("/fake/path", 384);
        assert!(emb.embed("hello").is_err());
        assert_eq!(emb.dimension(), 384);
    }

    #[test]
    fn api_stub_returns_error() {
        let emb = ApiEmbedder::new(
            "https://api.example.com",
            "key",
            1536,
            "text-embedding-3-small",
        );
        assert!(emb.embed("hello").is_err());
        assert_eq!(emb.model_name(), "text-embedding-3-small");
    }

    #[test]
    fn embedder_trait_object() {
        let emb: Box<dyn Embedder> = Box::new(MockEmbedder::new(64));
        let v = emb.embed("trait object test").unwrap();
        assert_eq!(v.len(), 64);
        assert_eq!(emb.dimension(), 64);
        assert_eq!(emb.model_name(), "mock-embedder");
    }
}
