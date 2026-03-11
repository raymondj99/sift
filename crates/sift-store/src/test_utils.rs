use crate::{FlatVectorIndex, MetadataStore};

pub fn create_test_flat_vector_store() -> FlatVectorIndex {
    FlatVectorIndex::new()
}

pub fn create_test_metadata_store() -> MetadataStore {
    MetadataStore::open_in_memory().unwrap()
}
