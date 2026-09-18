//! Adjacency types for differential privacy
//!
//! Defines the relationship between neighboring datasets in differential privacy analysis.

/// The type of adjacency relationship between two datasets
///
/// In differential privacy, we consider two datasets D and D' to be "neighbors"
/// if they differ in exactly one element. The adjacency type specifies the
/// relationship and affects the privacy loss distribution calculation.
///
/// Different adjacency types can result in different privacy loss distributions
/// for the same mechanism, particularly for asymmetric mechanisms like Poisson
/// subsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Adjacency {
    /// Dataset D has one fewer element than D' (D ⊂ D')
    Remove,
    /// Dataset D has one more element than D' (D' ⊂ D)
    Add,
    /// Datasets D and D' have the same size but differ in exactly one element
    Replace,
}

impl std::fmt::Display for Adjacency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjacency_display() {
        assert_eq!(format!("{}", Adjacency::Remove), "Remove");
        assert_eq!(format!("{}", Adjacency::Add), "Add");
        assert_eq!(format!("{}", Adjacency::Replace), "Replace");
    }
}
