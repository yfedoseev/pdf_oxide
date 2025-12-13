//! Logging utilities for the text extraction pipeline.
//!
//! This module provides convenient logging macros that respect the configured log level.
//!
//! When the `logging` feature is disabled, all logging is compiled out.
//! When enabled, logging goes to stderr with formatted messages.

/// Log an INFO level message.
///
/// Only logs when the configured log level is Info or more verbose.
/// Compiled out entirely when `logging` feature is disabled.
///
/// # Examples
///
/// ```ignore
/// extract_log_info!("Starting text extraction from 5 pages");
/// extract_log_info!("Processing page {}/{}", 1, 5);
/// ```
#[macro_export]
macro_rules! extract_log_info {
    ($msg:expr) => {
        #[cfg(feature = "logging")]
        eprintln!("[INFO] {}", $msg);
    };
    ($fmt:expr, $($arg:tt)*) => {
        #[cfg(feature = "logging")]
        eprintln!("[INFO] {}", format!($fmt, $($arg)*));
    };
}

/// Log a WARN level message.
///
/// Only logs when the configured log level is Warn or more verbose.
/// Compiled out entirely when `logging` feature is disabled.
///
/// # Examples
///
/// ```ignore
/// extract_log_warn!("Document contains unrecognized font");
/// extract_log_warn!("Page {} has invalid encoding", page_num);
/// ```
#[macro_export]
macro_rules! extract_log_warn {
    ($msg:expr) => {
        #[cfg(feature = "logging")]
        eprintln!("[WARN] {}", $msg);
    };
    ($fmt:expr, $($arg:tt)*) => {
        #[cfg(feature = "logging")]
        eprintln!("[WARN] {}", format!($fmt, $($arg)*));
    };
}

/// Log a DEBUG level message.
///
/// Only logs when the configured log level is Debug or more verbose.
/// Compiled out entirely when `logging` feature is disabled.
///
/// # Examples
///
/// ```ignore
/// extract_log_debug!("Processing page 1/5");
/// extract_log_debug!("Detected document script: {:?}", script);
/// ```
#[macro_export]
macro_rules! extract_log_debug {
    ($msg:expr) => {
        #[cfg(feature = "logging")]
        eprintln!("[DEBUG] {}", $msg);
    };
    ($fmt:expr, $($arg:tt)*) => {
        #[cfg(feature = "logging")]
        eprintln!("[DEBUG] {}", format!($fmt, $($arg)*));
    };
}

/// Log a TRACE level message.
///
/// Only logs when the configured log level is Trace.
/// Compiled out entirely when `logging` feature is disabled.
///
/// Use this for very detailed information like character-level details.
///
/// # Examples
///
/// ```ignore
/// extract_log_trace!("Detected 125 word boundaries on page 1");
/// extract_log_trace!("TJ offset: {:?}, threshold: {}, boundary: {}",
///     tj_offset, threshold, is_boundary);
/// ```
#[macro_export]
macro_rules! extract_log_trace {
    ($msg:expr) => {
        #[cfg(feature = "logging")]
        eprintln!("[TRACE] {}", $msg);
    };
    ($fmt:expr, $($arg:tt)*) => {
        #[cfg(feature = "logging")]
        eprintln!("[TRACE] {}", format!($fmt, $($arg)*));
    };
}

/// Log an ERROR level message.
///
/// Always logs (independent of log level configuration).
/// Compiled out entirely when `logging` feature is disabled.
///
/// # Examples
///
/// ```ignore
/// extract_log_error!("Failed to extract text from page");
/// extract_log_error!("Invalid PDF content: {}", error);
/// ```
#[macro_export]
macro_rules! extract_log_error {
    ($msg:expr) => {
        #[cfg(feature = "logging")]
        eprintln!("[ERROR] {}", $msg);
    };
    ($fmt:expr, $($arg:tt)*) => {
        #[cfg(feature = "logging")]
        eprintln!("[ERROR] {}", format!($fmt, $($arg)*));
    };
}

// Re-export the macros for convenience
pub use extract_log_debug;
pub use extract_log_error;
pub use extract_log_info;
pub use extract_log_trace;
pub use extract_log_warn;
