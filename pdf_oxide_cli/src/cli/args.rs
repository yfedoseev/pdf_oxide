use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "pdf-oxide",
    version,
    about = "Fast, local PDF processing",
    long_about = "pdf-oxide — the fastest PDF toolkit.\nRun with no arguments for interactive REPL mode."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Output file path (defaults to stdout for text outputs)
    #[arg(short, long, global = true)]
    pub output: Option<PathBuf>,

    /// Page range, e.g. "1-5", "1,3,7", "1-3,7,10-12"
    #[arg(short, long, global = true)]
    pub pages: Option<String>,

    /// Show verbose output with timing
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Suppress all non-essential output
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Output as JSON
    #[arg(short, long, global = true)]
    pub json: bool,

    /// Password for encrypted PDFs
    #[arg(long, global = true)]
    pub password: Option<String>,

    /// Skip the banner in REPL mode
    #[arg(long, global = true)]
    pub no_banner: bool,
}

#[derive(Subcommand)]
pub enum Command {
    /// Extract text from a PDF in various formats
    Text {
        /// Input PDF file
        file: PathBuf,

        /// Output format (plain, words, lines, structured)
        ///
        /// `structured` emits `StructuredPage` JSON per page — typed regions with
        /// `kind` (RegionRole), `column_index` and bbox — so two-column layouts
        /// come out as separate column blocks rather than line-interleaved.
        #[arg(long, value_parser = ["plain", "words", "lines", "structured"], default_value = "plain")]
        format: String,

        /// Specific area to extract from as x,y,width,height (points)
        #[arg(long)]
        area: Option<String>,
    },

    /// Classify each page (text vs OCR) — cheap preflight, no OCR.
    /// Prints JSON `DocumentClassification` (#517).
    Classify {
        /// Input PDF file
        file: PathBuf,
    },

    /// Auto-extract text: auto-routes text-vs-OCR per page with
    /// graceful native fallback (never the opaque OCR error — #513).
    Auto {
        /// Input PDF file
        file: PathBuf,

        /// Output format: `text` (assembled) or `json` (rich
        /// per-region `PageExtraction` with typed reasons)
        #[arg(long, value_parser = ["text", "json"], default_value = "text")]
        format: String,
    },

    /// Manage OCR/layout models (build-time provisioning, #513/#517)
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },

    /// Extract vector paths from a PDF
    Paths {
        /// Input PDF file
        file: PathBuf,

        /// Output format (json, rects, lines)
        #[arg(long, value_parser = ["json", "rects", "lines"], default_value = "json")]
        format: String,

        /// Specific area to extract from as x,y,width,height (points)
        #[arg(long)]
        area: Option<String>,
    },

    /// Convert PDF to Markdown
    Markdown {
        /// Input PDF file
        file: PathBuf,
    },

    /// Convert PDF to HTML
    Html {
        /// Input PDF file
        file: PathBuf,
    },

    /// Show PDF metadata and page count
    Info {
        /// Input PDF file
        file: PathBuf,
    },

    /// Merge multiple PDFs into one
    Merge {
        /// Input PDF files (first file is the base)
        #[arg(required = true, num_args = 2..)]
        files: Vec<PathBuf>,
    },

    /// Split a PDF into individual pages, or by bookmarks with --by-bookmarks
    Split {
        /// Input PDF file
        file: PathBuf,
        /// Split by document bookmarks/outline instead of per-page (#482)
        #[arg(long = "by-bookmarks")]
        by_bookmarks: bool,
        /// Only split at bookmarks whose title starts with this prefix
        #[arg(long = "bookmark-prefix", value_name = "PREFIX")]
        bookmark_prefix: Option<String>,
        /// Outline depth to split at (1 = top-level only, 0 = all levels)
        #[arg(long = "bookmark-level", default_value_t = 1)]
        bookmark_level: u32,
        /// Case-insensitive prefix matching for --bookmark-prefix
        #[arg(long = "ignore-case")]
        ignore_case: bool,
        /// Do not emit the pages before the first bookmark as a front-matter file
        #[arg(long = "no-front-matter")]
        no_front_matter: bool,
    },

    /// Create a PDF from Markdown, HTML, or plain text
    Create {
        /// Input source file
        file: PathBuf,

        /// Input format
        #[arg(long, value_parser = ["markdown", "html", "text"])]
        from: String,
    },

    /// Compress and optimize a PDF
    Compress {
        /// Input PDF file
        file: PathBuf,
    },

    /// Encrypt a PDF with a password (placeholder — coming in v0.4.0)
    Encrypt {
        /// Input PDF file
        file: PathBuf,
    },

    /// Decrypt a password-protected PDF
    Decrypt {
        /// Input PDF file
        file: PathBuf,

        /// Password to decrypt
        #[arg(long)]
        password: String,
    },

    /// Search for text in a PDF
    Search {
        /// Input PDF file
        file: PathBuf,

        /// Search pattern (regex supported)
        pattern: String,

        /// Case-insensitive search
        #[arg(short, long)]
        ignore_case: bool,
    },

    /// Extract images from a PDF
    Images {
        /// Input PDF file
        file: PathBuf,

        /// Specific area to extract from as x,y,width,height (points)
        #[arg(long)]
        area: Option<String>,
    },

    /// Rotate pages by 90, 180, or 270 degrees
    Rotate {
        /// Input PDF file
        file: PathBuf,

        /// Rotation angle in degrees (90, 180, 270, or -90)
        #[arg(long)]
        degrees: i32,
    },

    /// Remove specific pages from a PDF
    Delete {
        /// Input PDF file
        file: PathBuf,
    },

    /// Reorder pages in a PDF
    Reorder {
        /// Input PDF file
        file: PathBuf,

        /// New page order as comma-separated 1-indexed numbers (e.g. "3,1,2,5,4")
        #[arg(long)]
        order: String,
    },

    /// Read, edit, or strip PDF metadata
    Metadata {
        /// Input PDF file
        file: PathBuf,

        /// Set document title
        #[arg(long)]
        title: Option<String>,

        /// Set document author
        #[arg(long)]
        author: Option<String>,

        /// Set document subject
        #[arg(long)]
        subject: Option<String>,

        /// Set document keywords
        #[arg(long)]
        keywords: Option<String>,

        /// Strip all metadata fields
        #[arg(long)]
        strip: bool,
    },

    /// Add a text watermark to pages
    Watermark {
        /// Input PDF file
        file: PathBuf,

        /// Watermark text (presets: CONFIDENTIAL, DRAFT, SAMPLE, "DO NOT COPY")
        text: String,

        /// Opacity (0.0-1.0)
        #[arg(long, default_value = "0.3")]
        opacity: f32,

        /// Rotation angle in degrees
        #[arg(long, default_value = "45")]
        rotation: f32,

        /// Font size in points
        #[arg(long, default_value = "48")]
        font_size: f32,

        /// Text color as R,G,B (0.0-1.0 each, e.g. "0.8,0,0")
        #[arg(long)]
        color: Option<String>,
    },

    /// List document bookmarks/outline
    Bookmarks {
        /// Input PDF file
        file: PathBuf,
    },

    /// Flatten annotations and/or form fields
    Flatten {
        /// Input PDF file
        file: PathBuf,

        /// Flatten form fields
        #[arg(long)]
        forms: bool,

        /// Flatten annotations
        #[arg(long)]
        annotations: bool,
    },

    /// Destructively redact regions — true content removal (#231)
    Redact {
        /// Input PDF file
        file: PathBuf,

        /// Redaction rectangle as PAGE:x0,y0,x1,y1 (repeatable)
        #[arg(long = "rect")]
        rects: Vec<String>,

        /// Apply existing /Redact annotations in the source
        #[arg(long = "from-annotations")]
        from_annotations: bool,

        /// Overlay fill colour as R,G,B in 0..1 (default 0,0,0)
        #[arg(long)]
        fill: Option<String>,

        /// Do not scrub document metadata
        #[arg(long = "no-scrub-metadata")]
        no_scrub_metadata: bool,
    },

    /// Crop page margins
    Crop {
        /// Input PDF file
        file: PathBuf,

        /// Margins as left,right,top,bottom in points (e.g. "50,50,50,50")
        #[arg(long)]
        margins: String,
    },

    /// List, fill, or export form fields
    Forms {
        /// Input PDF file
        file: PathBuf,

        /// Fill fields as key=value pairs (e.g. "name=John,age=30")
        #[arg(long)]
        fill: Option<String>,

        /// Export form data (fdf or xfdf)
        #[arg(long, value_parser = ["fdf", "xfdf"])]
        export: Option<String>,

        /// Specific area to filter fields by as x,y,width,height (points)
        #[arg(long)]
        area: Option<String>,
    },

    /// Render PDF pages to images (PNG/JPEG)
    Render {
        /// Input PDF file
        file: PathBuf,

        /// DPI for rendering (default: 150)
        #[arg(long, default_value = "150")]
        dpi: u32,

        /// Output format (png or jpeg)
        #[arg(long, value_parser = ["png", "jpeg"], default_value = "png")]
        format: String,

        /// JPEG quality (1-100, only for jpeg)
        #[arg(long, default_value = "85")]
        quality: u8,
    },
}

/// `pdf-oxide models <action>` — build-time model provisioning (#517).
#[derive(Subcommand)]
pub enum ModelsAction {
    /// Download/verify OCR models into the cache dir
    /// (`$PDF_OXIDE_MODEL_DIR`). The documented Dockerfile `RUN`.
    /// Repeat `--language` for multi-language packs (default: english).
    /// e.g. `pdf-oxide models prefetch -l english -l chinese -l arabic`
    Prefetch {
        /// OCR language(s) to provision (english, chinese, arabic,
        /// cyrillic, latin, devanagari, korean, japanese,
        /// chinese_traditional, tamil, telugu, kannada). Default:
        /// english. Hebrew has no upstream PaddleOCR model.
        #[arg(short = 'l', long = "language", value_name = "LANG")]
        languages: Vec<String>,
        /// Provision EVERY supported language (the Docker/CI build
        /// case). Overrides `--language`. e.g. in a Dockerfile:
        /// `RUN pdf-oxide models prefetch --all`
        #[arg(long = "all", conflicts_with = "languages")]
        all: bool,
    },
    /// Print the JSON model manifest (`name/sha256/size/source_url`)
    /// for air-gapped verification.
    Manifest,
}
