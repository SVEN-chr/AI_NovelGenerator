use std::thread;
use std::time::Duration;

use log::warn;

use crate::error::AdapterError;

#[derive(Clone, Copy, Debug)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub sleep: Duration,
}

impl RetryConfig {
    pub const fn new(max_retries: usize, sleep: Duration) -> Self {
        Self { max_retries, sleep }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            sleep: Duration::from_secs(2),
        }
    }
}

pub fn call_with_retry<F, T>(mut f: F, config: &RetryConfig) -> Result<T, AdapterError>
where
    F: FnMut() -> Result<T, AdapterError>,
{
    let mut last_error: Option<AdapterError> = None;

    for attempt in 1..=config.max_retries {
        match f() {
            Ok(value) => return Ok(value),
            Err(err) => {
                let should_retry = attempt < config.max_retries;
                warn!(
                    "[call_with_retry] attempt {}/{} failed: {}",
                    attempt, config.max_retries, err
                );
                if should_retry {
                    thread::sleep(config.sleep);
                }
                last_error = Some(err);
            }
        }
    }

    let err = last_error.unwrap_or(AdapterError::EmptyResponse);
    Err(AdapterError::retry_exhausted(config.max_retries, err))
}
