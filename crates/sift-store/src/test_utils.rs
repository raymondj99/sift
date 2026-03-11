use crate::{FlatVectorIndex, MetadataStore};

pub fn create_test_flat_vector_store() -> FlatVectorIndex {
    FlatVectorIndex::new()
}

pub fn create_test_metadata_store() -> MetadataStore {
    MetadataStore::open_in_memory().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::VectorStore;

    #[test]
    fn test_create_test_flat_vector_store() {
        let store = create_test_flat_vector_store();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_create_test_metadata_store() {
        let store = create_test_metadata_store();
        let stats = store.stats().unwrap();
        assert_eq!(stats.total_sources, 0);
    }
}
