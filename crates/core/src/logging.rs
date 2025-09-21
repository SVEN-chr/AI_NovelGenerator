use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        };
        f.write_str(label)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LogRecord {
    pub level: LogLevel,
    pub message: String,
}

impl LogRecord {
    pub fn new(level: LogLevel, message: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
        }
    }
}

pub trait LogSink: Send + Sync {
    fn log(&self, record: LogRecord);
}

pub type SharedLogSink = Arc<dyn LogSink>;

#[derive(Default)]
pub struct NullLogSink;

impl LogSink for NullLogSink {
    fn log(&self, _record: LogRecord) {}
}

#[derive(Default)]
pub struct VecLogSink {
    records: Mutex<Vec<LogRecord>>,
}

impl VecLogSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, record: LogRecord) {
        if let Ok(mut guard) = self.records.lock() {
            guard.push(record);
        }
    }

    pub fn records(&self) -> Vec<LogRecord> {
        self.records
            .lock()
            .map(|guard| guard.iter().cloned().collect())
            .unwrap_or_default()
    }
}

impl LogSink for VecLogSink {
    fn log(&self, record: LogRecord) {
        self.push(record);
    }
}

#[derive(Default, Clone)]
pub struct StdoutLogSink;

impl StdoutLogSink {
    pub fn new() -> Self {
        Self
    }
}

impl LogSink for StdoutLogSink {
    fn log(&self, record: LogRecord) {
        println!("[{}] {}", record.level, record.message);
    }
}
