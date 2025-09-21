use std::error::Error as StdError;
use std::fmt;

#[derive(Debug)]
pub struct EmbeddingModelError {
    inner: Box<dyn StdError + Send + Sync>,
}

impl EmbeddingModelError {
    pub fn new<E>(error: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self {
            inner: Box::new(error),
        }
    }

    pub fn into_inner(self) -> Box<dyn StdError + Send + Sync> {
        self.inner
    }

    pub fn as_inner(&self) -> &(dyn StdError + Send + Sync + 'static) {
        self.inner.as_ref()
    }
}

impl fmt::Display for EmbeddingModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl StdError for EmbeddingModelError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.inner.as_ref())
    }
}

pub trait EmbeddingModel: Send + Sync {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingModelError>;

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingModelError>;
}
