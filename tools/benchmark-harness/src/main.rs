//! pdf_oxide extraction-quality benchmark.
//!
//! Computes TF1 (token F1) and SF1 (block-weighted structural F1 with
//! LIS order penalty) against a directory of ground-truth markdown files.
//! See `PLAN.md` for scoring formulas and sequencing.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod consensus;
mod engine;
mod report;
mod score;
mod sf1;

#[derive(Parser)]
#[command(name = "benchmark-harness", version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run an engine against a corpus and emit a JSON report.
    Run(RunArgs),
    /// Compare two JSON reports; exit non-zero on meaningful regression.
    Diff(DiffArgs),
}

#[derive(Parser)]
pub struct RunArgs {
    /// Engine to benchmark.
    #[arg(long, value_enum)]
    pub engine: engine::EngineKind,

    /// Directory containing PDFs to extract.
    #[arg(long)]
    pub corpus: PathBuf,

    /// Directory of ground-truth markdown files, matched by stem.
    /// If omitted, `--consensus-peers` must be set to generate a
    /// pseudo-reference from peer engines.
    #[arg(long, required_unless_present = "consensus_peers")]
    pub ground_truth: Option<PathBuf>,

    /// Comma-separated list of peer engines whose intersection is
    /// used as pseudo-ground-truth. Example: `--consensus-peers
    /// pdftotext,pdfium`. Scoring labels `reference=consensus`.
    #[arg(long, value_delimiter = ',')]
    pub consensus_peers: Vec<engine::EngineKind>,

    /// Minimum peer agreement count when `--consensus-peers` is set.
    #[arg(long, default_value_t = 2)]
    pub consensus_min_agree: usize,

    /// Output JSON report path.
    #[arg(long)]
    pub output: PathBuf,

    /// Seconds before an individual extraction is aborted (0 = no limit).
    #[arg(long, default_value_t = 60)]
    pub timeout_secs: u64,
}

#[derive(Parser)]
pub struct DiffArgs {
    pub base: PathBuf,
    pub head: PathBuf,

    /// Fail if mean TF1 drops by more than this (percentage points).
    #[arg(long, default_value_t = 0.5)]
    pub mean_tf1_drop_pp: f64,

    /// Fail if any fixture's TF1 drops by more than this (pp).
    #[arg(long, default_value_t = 5.0)]
    pub per_fixture_tf1_drop_pp: f64,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run(args) => report::run(args),
        Cmd::Diff(args) => report::diff(args),
    }
}
