//! Engine adapters.
//!
//! Each engine extracts a PDF to markdown. The trait carries a `name()`
//! and a single `extract` method so new adapters (docling, marker, …)
//! only need one file and one enum arm.

use anyhow::{anyhow, Context, Result};
use clap::ValueEnum;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum EngineKind {
    PdfOxide,
    Pdftotext,
    #[cfg(feature = "pdfium")]
    Pdfium,
}

pub struct Extraction {
    pub markdown: String,
    pub duration: Duration,
}

pub trait Engine {
    fn name(&self) -> &'static str;
    fn extract(&self, pdf: &Path) -> Result<Extraction>;
}

pub fn build(kind: EngineKind) -> Result<Box<dyn Engine>> {
    Ok(match kind {
        EngineKind::PdfOxide => Box::new(PdfOxideEngine),
        EngineKind::Pdftotext => Box::new(PdftotextEngine::new()?),
        #[cfg(feature = "pdfium")]
        EngineKind::Pdfium => Box::new(PdfiumEngine::new()?),
    })
}

// ─── pdf_oxide (in-process) ───────────────────────────────────────────────

pub struct PdfOxideEngine;

impl Engine for PdfOxideEngine {
    fn name(&self) -> &'static str {
        "pdf_oxide"
    }

    fn extract(&self, pdf: &Path) -> Result<Extraction> {
        use pdf_oxide::PdfDocument;
        let start = Instant::now();
        let mut doc = PdfDocument::open(pdf).with_context(|| format!("open {}", pdf.display()))?;
        let page_count = doc.page_count().unwrap_or(0);
        let mut md = String::new();
        for page in 0..page_count {
            // Text-only for now. When the markdown converter stabilises we
            // swap to it so SF1 can score block structure for pdf_oxide.
            let Ok(text) = doc.extract_text(page) else {
                continue;
            };
            md.push_str(&text);
            md.push('\n');
        }
        Ok(Extraction {
            markdown: md,
            duration: start.elapsed(),
        })
    }
}

// ─── pdftotext (poppler subprocess) ───────────────────────────────────────

/// Wraps the `pdftotext` binary from poppler-utils. Emits plain text (not
/// markdown) — SF1 will score low on structure for this engine, which is
/// accurate: pdftotext makes no structure claim. TF1 is the meaningful
/// metric here.
pub struct PdftotextEngine {
    bin: String,
}

impl PdftotextEngine {
    pub fn new() -> Result<Self> {
        // Allow override (e.g. for non-standard install locations).
        let bin = std::env::var("PDFTOTEXT_BIN").unwrap_or_else(|_| "pdftotext".to_string());
        // Probe once so a missing binary fails fast, not per fixture.
        let status = Command::new(&bin).arg("-v").output();
        if status.is_err() {
            return Err(anyhow!(
                "pdftotext not found at `{bin}` — install poppler-utils or \
                 set PDFTOTEXT_BIN=/path/to/pdftotext"
            ));
        }
        Ok(Self { bin })
    }
}

impl Engine for PdftotextEngine {
    fn name(&self) -> &'static str {
        "pdftotext"
    }

    fn extract(&self, pdf: &Path) -> Result<Extraction> {
        let start = Instant::now();
        let output = Command::new(&self.bin)
            .args(["-layout", "-enc", "UTF-8"])
            .arg(pdf)
            .arg("-") // stdout
            .output()
            .with_context(|| format!("invoke {} on {}", self.bin, pdf.display()))?;
        if !output.status.success() {
            return Err(anyhow!(
                "pdftotext failed on {}: {}",
                pdf.display(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(Extraction {
            markdown: String::from_utf8_lossy(&output.stdout).into_owned(),
            duration: start.elapsed(),
        })
    }
}

// ─── pdfium (Chrome's PDF engine via pdfium-render) ────────────────────────

#[cfg(feature = "pdfium")]
pub struct PdfiumEngine {
    pdfium: pdfium_render::prelude::Pdfium,
}

#[cfg(feature = "pdfium")]
impl PdfiumEngine {
    pub fn new() -> Result<Self> {
        use pdfium_render::prelude::Pdfium;
        // Try the system library first, fall back to a bundled copy at
        // $PDFIUM_DYNAMIC_LIB_PATH. The crate's bind_to_library API returns
        // a descriptive error when the .so/.dylib is missing.
        let bindings = match std::env::var("PDFIUM_DYNAMIC_LIB_PATH") {
            Ok(path) => {
                Pdfium::bind_to_library(path).context("load pdfium from PDFIUM_DYNAMIC_LIB_PATH")?
            },
            Err(_) => Pdfium::bind_to_system_library()
                .context("pdfium system library not found; set PDFIUM_DYNAMIC_LIB_PATH")?,
        };
        Ok(Self {
            pdfium: Pdfium::new(bindings),
        })
    }
}

#[cfg(feature = "pdfium")]
impl Engine for PdfiumEngine {
    fn name(&self) -> &'static str {
        "pdfium"
    }

    fn extract(&self, pdf: &Path) -> Result<Extraction> {
        let start = Instant::now();
        let document = self
            .pdfium
            .load_pdf_from_file(pdf, None)
            .with_context(|| format!("pdfium load {}", pdf.display()))?;
        let mut md = String::new();
        for page in document.pages().iter() {
            let text = page.text().map_err(|e| anyhow!("pdfium page text: {e}"))?;
            md.push_str(&text.all());
            md.push('\n');
        }
        Ok(Extraction {
            markdown: md,
            duration: start.elapsed(),
        })
    }
}
