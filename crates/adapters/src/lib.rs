mod base_url;
mod embedding;
mod error;
mod llm;
mod retry;
mod vectorstore;

pub use base_url::{check_base_url, ensure_openai_base_url_has_v1};
pub use embedding::{create_embedding_adapter, create_embedding_adapter_from_profile};
pub use error::AdapterError;
pub use llm::{create_llm_adapter, create_llm_adapter_from_profile};
pub use retry::{call_with_retry, RetryConfig};
pub use vectorstore::{
    get_vectorstore_dir, import_knowledge_file, init_vector_store, load_vector_store,
    similarity_search, update_vector_store, QdrantVectorStore, VectorStoreConfig,
};

pub use novel_core::architecture::{LanguageModel, LanguageModelError};
pub use novel_core::config::{Config, ConfigStore, EmbeddingConfig, LlmConfig, NovelConfig};
pub use novel_core::embedding::{EmbeddingModel, EmbeddingModelError};
