use std::io;
use std::path::Path;

use novel_core::embedding::EmbeddingModelError;
use reqwest::StatusCode;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("failed to parse response: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid adapter configuration: {0}")]
    InvalidConfig(String),
    #[error("unexpected http status {status}: {body}")]
    HttpStatus { status: StatusCode, body: String },
    #[error("io error at `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("embedding operation failed: {0}")]
    Embedding(#[from] EmbeddingModelError),
    #[error("vector store operation failed: {0}")]
    VectorStore(String),
    #[error("operation failed after {attempts} attempts: {source}")]
    RetryExhausted {
        attempts: usize,
        #[source]
        source: Box<AdapterError>,
    },
    #[error("API returned an empty response")]
    EmptyResponse,
}

impl AdapterError {
    pub fn retry_exhausted(attempts: usize, source: AdapterError) -> Self {
        AdapterError::RetryExhausted {
            attempts,
            source: Box::new(source),
        }
    }

    pub fn io(path: &Path, source: io::Error) -> Self {
        AdapterError::Io {
            path: path.display().to_string(),
            source,
        }
    }
}
