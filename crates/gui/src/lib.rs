//! GUI crate placeholder that depends on the shared core utilities.

pub use novel_adapters::{
    import_knowledge_file, load_vector_store, similarity_search, update_vector_store,
    QdrantVectorStore, VectorStoreConfig,
};
pub use novel_core::logging::{LogLevel, LogRecord, LogSink};
pub use novel_core::vectorstore::{split_text_segments, DEFAULT_SEGMENT_CHAR_LIMIT};
