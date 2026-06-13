//! CLI interface for pdf-oxide.
//!
//! Provides both subcommand execution and interactive REPL modes.

mod args;
mod banner;
mod colors;
pub mod commands;
mod pages;
mod repl;

use args::{Cli, Command};
use clap::Parser;
use std::path::Path;

/// Run the CLI. Called from the `pdf-oxide` binary entry point.
pub fn run() -> pdf_oxide::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();

    if args.len() <= 1 {
        if is_terminal::is_terminal(std::io::stdin()) {
            return repl::enter(false, None, false, false);
        } else {
            return run_piped_stdin();
        }
    }

    let cli = Cli::parse();

    match cli.command {
        Some(cmd) => dispatch(
            cmd,
            cli.output.as_deref(),
            cli.pages.as_deref(),
            cli.password.as_deref(),
            cli.verbose,
            cli.quiet,
            cli.json,
        ),
        None => repl::enter(cli.no_banner, cli.password, cli.json, cli.verbose),
    }
}

fn dispatch(
    cmd: Command,
    output: Option<&Path>,
    pages: Option<&str>,
    password: Option<&str>,
    verbose: bool,
    quiet: bool,
    json: bool,
) -> pdf_oxide::Result<()> {
    let start = if verbose {
        Some(std::time::Instant::now())
    } else {
        None
    };

    let result = match cmd {
        Command::Text {
            ref file,
            ref format,
            ref column_mode,
            ref area,
        } => commands::text::run(
            file,
            format,
            column_mode,
            area.as_deref(),
            pages,
            output,
            password,
            json,
        ),
        Command::Paths {
            ref file,
            ref format,
            ref area,
        } => commands::paths::run(file, format, area.as_deref(), pages, output, password, json),
        Command::Markdown { ref file } => {
            commands::markdown::run(file, pages, output, password, json)
        },
        Command::Html { ref file } => commands::html::run(file, pages, output, password, json),
        Command::Classify { ref file } => commands::classify::run(file, password, json),
        Command::Auto {
            ref file,
            ref format,
        } => commands::auto::run(file, format, pages, output, password, json),
        Command::Models { ref action } => commands::models::run(action),
        Command::Info { ref file } => commands::info::run(file, password, json),
        Command::Merge { ref files } => commands::merge::run(files, output),
        Command::Split {
            ref file,
            by_bookmarks,
            ref bookmark_prefix,
            bookmark_level,
            ignore_case,
            no_front_matter,
        } => commands::split::run(
            file,
            pages,
            output,
            password,
            by_bookmarks,
            bookmark_prefix.as_deref(),
            bookmark_level,
            ignore_case,
            no_front_matter,
        ),
        Command::Create { ref file, ref from } => commands::create::run(file, from, output),
        Command::Compress { ref file } => commands::compress::run(file, output, password),
        Command::Encrypt { .. } => commands::encrypt::run(),
        Command::Decrypt {
            ref file,
            ref password,
        } => commands::decrypt::run(file, password, output),
        Command::Search {
            ref file,
            ref pattern,
            ignore_case,
        } => commands::search::run(file, pattern, ignore_case, pages, password, json),
        Command::Images { ref file, ref area } => {
            commands::images::run(file, area.as_deref(), pages, output, password, json)
        },
        Command::Rotate { ref file, degrees } => {
            commands::rotate::run(file, degrees, pages, output, password)
        },
        Command::Delete { ref file } => commands::delete::run(file, pages, output, password),
        Command::Reorder {
            ref file,
            ref order,
        } => commands::reorder::run(file, order, output, password),
        Command::Metadata {
            ref file,
            ref title,
            ref author,
            ref subject,
            ref keywords,
            strip,
        } => commands::metadata::run(
            file,
            title.as_deref(),
            author.as_deref(),
            subject.as_deref(),
            keywords.as_deref(),
            strip,
            output,
            password,
            json,
        ),
        Command::Watermark {
            ref file,
            ref text,
            opacity,
            rotation,
            font_size,
            ref color,
        } => commands::watermark::run(
            file,
            text,
            opacity,
            rotation,
            font_size,
            color.as_deref(),
            pages,
            output,
            password,
        ),
        Command::Bookmarks { ref file } => commands::bookmarks::run(file, password, json),
        Command::Flatten {
            ref file,
            forms,
            annotations,
        } => commands::flatten::run(file, forms, annotations, output, password),
        Command::Redact {
            ref file,
            ref rects,
            from_annotations,
            ref fill,
            no_scrub_metadata,
        } => commands::redact::run(
            file,
            rects,
            from_annotations,
            fill.as_deref(),
            no_scrub_metadata,
            output,
            password,
        ),
        Command::Crop {
            ref file,
            ref margins,
        } => commands::crop::run(file, margins, pages, output, password),
        Command::Forms {
            ref file,
            ref fill,
            ref export,
            ref area,
        } => commands::forms::run(
            file,
            fill.as_deref(),
            export.as_deref(),
            area.as_deref(),
            pages,
            output,
            password,
            json,
        ),
        Command::Render {
            ref file,
            dpi,
            ref format,
            quality,
        } => commands::render::run(file, dpi, format, quality, pages, output, password),
    };

    if let Some(start) = start {
        let elapsed = start.elapsed();
        if !quiet {
            eprintln!("Completed in {:.1}ms", elapsed.as_secs_f64() * 1000.0);
        }
    }

    result
}

fn run_piped_stdin() -> pdf_oxide::Result<()> {
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let reader = stdin.lock();

    if let Some(Ok(line)) = reader.lines().next() {
        let path = line.trim().to_string();
        if path.is_empty() {
            return Err(pdf_oxide::Error::InvalidOperation(
                "No file path provided on stdin".to_string(),
            ));
        }
        let file = std::path::PathBuf::from(&path);
        commands::text::run(&file, "plain", "auto", None, None, None, None, false)
    } else {
        Err(pdf_oxide::Error::InvalidOperation("No input received on stdin".to_string()))
    }
}
