//! Concrete `VectorSpace` implementation for dense f32 vectors.

use crate::traits::VectorSpace;

/// A dense f32 embedding vector implementing `VectorSpace`.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseVector {
    data: Vec<f32>,
}

impl DenseVector {
    /// Create from an existing Vec.
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Create from a slice.
    pub fn from_slice(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
}

impl VectorSpace for DenseVector {
    fn dim(&self) -> usize {
        self.data.len()
    }

    fn zero(dim: usize) -> Self {
        Self {
            data: vec![0.0; dim],
        }
    }

    fn add(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len(), "dimension mismatch");
        Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    fn scale(&self, factor: f32) -> Self {
        Self {
            data: self.data.iter().map(|v| v * factor).collect(),
        }
    }

    fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

impl From<Vec<f32>> for DenseVector {
    fn from(data: Vec<f32>) -> Self {
        Self::new(data)
    }
}

impl From<&[f32]> for DenseVector {
    fn from(data: &[f32]) -> Self {
        Self::from_slice(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_vector() {
        let v = DenseVector::zero(3);
        assert_eq!(v.dim(), 3);
        assert_eq!(v.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn add() {
        let a = DenseVector::new(vec![1.0, 2.0, 3.0]);
        let b = DenseVector::new(vec![4.0, 5.0, 6.0]);
        let c = a.add(&b);
        assert_eq!(c.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn scale() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
        let scaled = v.scale(2.0);
        assert_eq!(scaled.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn scale_zero() {
        let v = DenseVector::new(vec![1.0, 2.0]);
        let z = v.scale(0.0);
        assert_eq!(z.as_slice(), &[0.0, 0.0]);
    }

    #[test]
    fn add_to_zero_is_identity() {
        let v = DenseVector::new(vec![3.0, 4.0]);
        let z = DenseVector::zero(2);
        assert_eq!(v.add(&z), v);
    }

    #[test]
    fn from_vec() {
        let v: DenseVector = vec![1.0, 2.0].into();
        assert_eq!(v.dim(), 2);
    }

    #[test]
    fn from_slice() {
        let data = [1.0f32, 2.0, 3.0];
        let v: DenseVector = data.as_slice().into();
        assert_eq!(v.dim(), 3);
    }

    #[test]
    fn clone_is_independent() {
        let v = DenseVector::new(vec![1.0, 2.0]);
        let v2 = v.clone();
        assert_eq!(v, v2);
    }
}
