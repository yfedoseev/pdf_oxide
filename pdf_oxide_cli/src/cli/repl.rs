use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use super::colors;
use pdf_oxide::PdfDocument;

struct ReplState {
    current_doc: Option<PdfDocument>,
    current_file: Option<PathBuf>,
    password: Option<String>,
    json: bool,
}

impl ReplState {
    fn prompt(&self) -> String {
        if let Some(ref f) = self.current_file {
            let name = f.file_name().and_then(|s| s.to_str()).unwrap_or("?");
            format!("pdf-oxide [{}]> ", name)
        } else {
            "pdf-oxide> ".to_string()
        }
    }

    fn ensure_doc(&mut self) -> pdf_oxide::Result<&mut PdfDocument> {
        self.current_doc.as_mut().ok_or_else(|| {
            pdf_oxide::Error::InvalidOperation(
                "No PDF loaded. Use 'open <file>' first.".to_string(),
            )
        })
    }
}

pub fn enter(
    no_banner: bool,
    password: Option<String>,
    json: bool,
    _verbose: bool,
) -> pdf_oxide::Result<()> {
    if !no_banner {
        super::banner::print_banner();
        eprintln!("Type {} for commands, {} to quit.", colors::bold("help"), colors::bold("exit"));
        eprintln!();
    }

    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut state = ReplState {
        current_doc: None,
        current_file: None,
        password,
        json,
    };
    let mut line = String::new();

    loop {
        eprint!("{}", colors::rust_orange(&state.prompt()));
        std::io::stderr().flush().ok();

        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break, // EOF (Ctrl+D)
            Ok(_) => {},
            Err(e) => {
                eprintln!("{}", colors::error(&format!("Read error: {e}")));
                break;
            },
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.splitn(2, char::is_whitespace).collect();
        let cmd = parts[0].to_lowercase();
        let args = parts.get(1).map(|s| s.trim()).unwrap_or("");

        let result = match cmd.as_str() {
            "exit" | "quit" | "q" | "bye" => break,
            "help" | "?" | "h" => {
                print_help();
                Ok(())
            },
            "open" | "o" | "load" => cmd_open(&mut state, args),
            "close" | "c" => cmd_close(&mut state),
            "text" | "t" => cmd_text(&mut state, args),
            "markdown" | "md" => cmd_markdown(&mut state, args),
            "html" => cmd_html(&mut state, args),
            "info" | "i" => cmd_info(&mut state, args),
            "search" | "s" | "find" | "grep" => cmd_search(&mut state, args),
            "images" | "img" => cmd_images(&mut state, args),
            "pages" | "p" => cmd_pages(&mut state),
            "bookmarks" | "bm" | "outline" | "toc" => cmd_bookmarks(&mut state, args),
            "forms" | "fields" => cmd_forms(&mut state, args),
            "rotate" => cmd_rotate(&mut state, args),
            "delete" | "del" | "rm" => cmd_delete(&mut state, args),
            "reorder" => cmd_reorder(&mut state, args),
            "metadata" | "meta" => cmd_metadata(&mut state, args),
            "watermark" | "wm" => cmd_watermark(&mut state, args),
            "flatten" => cmd_flatten(&mut state, args),
            "crop" => cmd_crop(&mut state, args),
            _ => {
                eprintln!("Unknown command: '{}'. Type 'help' for available commands.", cmd);
                Ok(())
            },
        };

        if let Err(e) = result {
            eprintln!("{}", colors::error(&format!("Error: {e}")));
        }
    }

    Ok(())
}

fn print_help() {
    eprintln!("Commands:");
    eprintln!("  open|o <file>      Load a PDF file");
    eprintln!("  close|c            Close the current PDF");
    eprintln!("  info|i [file]      Show PDF metadata");
    eprintln!("  text|t [file]      Extract plain text");
    eprintln!("  markdown|md [file] Convert to Markdown");
    eprintln!("  html [file]        Convert to HTML");
    eprintln!("  search|s <pattern> Search text (also: find, grep)");
    eprintln!("  images|img [file]  Extract images to current directory");
    eprintln!("  pages|p            Show page count of current PDF");
    eprintln!("  bookmarks|bm [file] List document bookmarks/outline (also: outline, toc)");
    eprintln!("  forms|fields [file] List form fields");
    eprintln!();
    eprintln!("Editing:");
    eprintln!("  rotate <degrees> [-o out.pdf] [--pages 1-3]  Rotate pages (90/180/270/-90)");
    eprintln!("  delete --pages 2,5-7 [-o out.pdf]            Remove pages");
    eprintln!("  reorder <3,1,2> [-o out.pdf]                 Reorder pages");
    eprintln!("  metadata [--title ...] [--author ...] [--strip] [-o out.pdf]  Edit metadata");
    eprintln!("  watermark <text> [-o out.pdf] [--pages 1-3]  Add watermark");
    eprintln!("  flatten [--forms] [--annotations] [-o out.pdf]  Flatten annotations/forms");
    eprintln!("  crop <l,r,t,b> [-o out.pdf] [--pages 1-3]   Crop margins");
    eprintln!();
    eprintln!("  help|h|?           Show this help message");
    eprintln!("  exit|quit|q        Exit the REPL (also: bye, Ctrl+D)");
}

fn cmd_open(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    if args.is_empty() {
        return Err(pdf_oxide::Error::InvalidOperation("Usage: open <file>".to_string()));
    }
    let path = PathBuf::from(args);
    let doc = PdfDocument::open(&path)?;
    if let Some(ref pw) = state.password {
        doc.authenticate(pw.as_bytes())?;
    }
    let pages = doc.page_count()?;
    state.current_doc = Some(doc);
    state.current_file = Some(path.clone());
    eprintln!("Opened {} ({} pages)", path.display(), pages);
    Ok(())
}

fn cmd_close(state: &mut ReplState) -> pdf_oxide::Result<()> {
    if state.current_doc.is_some() {
        let name = state
            .current_file
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        state.current_doc = None;
        state.current_file = None;
        eprintln!("Closed {name}");
    } else {
        eprintln!("No PDF is currently open.");
    }
    Ok(())
}

fn with_doc(
    state: &mut ReplState,
    args: &str,
    f: impl FnOnce(&mut PdfDocument) -> pdf_oxide::Result<()>,
) -> pdf_oxide::Result<()> {
    if args.is_empty() {
        let doc = state.ensure_doc()?;
        f(doc)
    } else {
        let mut doc = PdfDocument::open(args)?;
        if let Some(ref pw) = state.password {
            doc.authenticate(pw.as_bytes())?;
        }
        f(&mut doc)
    }
}

fn cmd_text(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    if !args.is_empty() {
        super::commands::text::run(
            Path::new(args),
            "plain",
            "auto",
            None,
            None,
            None,
            state.password.as_deref(),
            state.json,
        )
    } else {
        let path = state
            .current_file
            .as_ref()
            .ok_or_else(|| {
                pdf_oxide::Error::InvalidOperation(
                    "No PDF loaded. Use 'open <file>' or provide a file path.".to_string(),
                )
            })?
            .clone();
        super::commands::text::run(
            &path,
            "plain",
            "auto",
            None,
            None,
            None,
            state.password.as_deref(),
            state.json,
        )
    }
}

fn cmd_markdown(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    with_doc(state, args, |doc| {
        let page_count = doc.page_count()?;
        let options = pdf_oxide::converters::ConversionOptions::default();
        for i in 0..page_count {
            let md = doc.to_markdown(i, &options)?;
            println!("{md}");
        }
        Ok(())
    })
}

fn cmd_html(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    with_doc(state, args, |doc| {
        let page_count = doc.page_count()?;
        let options = pdf_oxide::converters::ConversionOptions::default();
        for i in 0..page_count {
            let html = doc.to_html(i, &options)?;
            println!("{html}");
        }
        Ok(())
    })
}

fn cmd_info(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    if !args.is_empty() {
        super::commands::info::run(Path::new(args), state.password.as_deref(), state.json)
    } else {
        let path = state
            .current_file
            .as_ref()
            .ok_or_else(|| {
                pdf_oxide::Error::InvalidOperation(
                    "No PDF loaded. Use 'open <file>' or provide a file path.".to_string(),
                )
            })?
            .clone();
        super::commands::info::run(&path, state.password.as_deref(), state.json)
    }
}

fn cmd_search(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    if args.is_empty() {
        return Err(pdf_oxide::Error::InvalidOperation("Usage: search <pattern>".to_string()));
    }
    let doc = state.ensure_doc()?;
    let options = pdf_oxide::search::SearchOptions::default();
    let results = pdf_oxide::search::TextSearcher::search(doc, args, &options)?;

    if results.is_empty() {
        eprintln!("No matches found for '{args}'");
    } else {
        eprintln!("Found {} match(es):", results.len());
        for r in &results {
            println!("  Page {}: \"{}\"", r.page + 1, r.text);
        }
    }
    Ok(())
}

fn cmd_images(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    if !args.is_empty() {
        super::commands::images::run(
            Path::new(args),
            None,
            None,
            Some(Path::new(".")),
            state.password.as_deref(),
            state.json,
        )
    } else {
        let path = state
            .current_file
            .as_ref()
            .ok_or_else(|| {
                pdf_oxide::Error::InvalidOperation(
                    "No PDF loaded. Use 'open <file>' or provide a file path.".to_string(),
                )
            })?
            .clone();
        super::commands::images::run(
            &path,
            None,
            None,
            Some(Path::new(".")),
            state.password.as_deref(),
            state.json,
        )
    }
}

fn cmd_pages(state: &mut ReplState) -> pdf_oxide::Result<()> {
    let doc = state.ensure_doc()?;
    let count = doc.page_count()?;
    println!("{count} pages");
    Ok(())
}

fn cmd_bookmarks(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    if !args.is_empty() {
        super::commands::bookmarks::run(Path::new(args), state.password.as_deref(), state.json)
    } else {
        let path = state
            .current_file
            .as_ref()
            .ok_or_else(|| {
                pdf_oxide::Error::InvalidOperation(
                    "No PDF loaded. Use 'open <file>' or provide a file path.".to_string(),
                )
            })?
            .clone();
        super::commands::bookmarks::run(&path, state.password.as_deref(), state.json)
    }
}

fn cmd_forms(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    if !args.is_empty() {
        super::commands::forms::run(
            Path::new(args),
            None,
            None,
            None,
            None,
            None,
            state.password.as_deref(),
            state.json,
        )
    } else {
        let path = state
            .current_file
            .as_ref()
            .ok_or_else(|| {
                pdf_oxide::Error::InvalidOperation(
                    "No PDF loaded. Use 'open <file>' or provide a file path.".to_string(),
                )
            })?
            .clone();
        super::commands::forms::run(
            &path,
            None,
            None,
            None,
            None,
            None,
            state.password.as_deref(),
            state.json,
        )
    }
}

/// Helper to get the current file path or error.
fn require_file(state: &ReplState) -> pdf_oxide::Result<PathBuf> {
    state.current_file.clone().ok_or_else(|| {
        pdf_oxide::Error::InvalidOperation("No PDF loaded. Use 'open <file>' first.".to_string())
    })
}

/// Simple flag parser for REPL args. Returns (positional_args, flags_map).
fn parse_repl_args(args: &str) -> (Vec<String>, std::collections::HashMap<String, String>) {
    let mut positional = Vec::new();
    let mut flags = std::collections::HashMap::new();
    let tokens: Vec<&str> = args.split_whitespace().collect();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i].starts_with("--") || tokens[i] == "-o" {
            let key = tokens[i].trim_start_matches('-').to_string();
            if i + 1 < tokens.len() && !tokens[i + 1].starts_with('-') {
                flags.insert(key, tokens[i + 1].to_string());
                i += 2;
            } else {
                flags.insert(key, String::new());
                i += 1;
            }
        } else {
            positional.push(tokens[i].to_string());
            i += 1;
        }
    }
    (positional, flags)
}

fn cmd_rotate(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    let path = require_file(state)?;
    let (positional, flags) = parse_repl_args(args);

    let degrees: i32 = positional
        .first()
        .ok_or_else(|| {
            pdf_oxide::Error::InvalidOperation(
                "Usage: rotate <degrees> [-o out.pdf] [--pages 1-3]".to_string(),
            )
        })?
        .parse()
        .map_err(|_| {
            pdf_oxide::Error::InvalidOperation(
                "Degrees must be a number (90, 180, 270, -90)".to_string(),
            )
        })?;

    let output = flags.get("o").map(PathBuf::from);
    let pages = flags.get("pages").map(|s| s.as_str());

    super::commands::rotate::run(
        &path,
        degrees,
        pages,
        output.as_deref(),
        state.password.as_deref(),
    )
}

fn cmd_delete(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    let path = require_file(state)?;
    let (_positional, flags) = parse_repl_args(args);

    let pages = flags.get("pages").map(|s| s.as_str());
    let output = flags.get("o").map(PathBuf::from);

    super::commands::delete::run(&path, pages, output.as_deref(), state.password.as_deref())
}

fn cmd_reorder(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    let path = require_file(state)?;
    let (positional, flags) = parse_repl_args(args);

    let order = positional.first().ok_or_else(|| {
        pdf_oxide::Error::InvalidOperation("Usage: reorder <3,1,2,5,4> [-o out.pdf]".to_string())
    })?;

    let output = flags.get("o").map(PathBuf::from);

    super::commands::reorder::run(&path, order, output.as_deref(), state.password.as_deref())
}

fn cmd_metadata(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    let path = require_file(state)?;
    let (_positional, flags) = parse_repl_args(args);

    let title = flags.get("title").map(|s| s.as_str());
    let author = flags.get("author").map(|s| s.as_str());
    let subject = flags.get("subject").map(|s| s.as_str());
    let keywords = flags.get("keywords").map(|s| s.as_str());
    let strip = flags.contains_key("strip");
    let output = flags.get("o").map(PathBuf::from);

    super::commands::metadata::run(
        &path,
        title,
        author,
        subject,
        keywords,
        strip,
        output.as_deref(),
        state.password.as_deref(),
        state.json,
    )
}

fn cmd_watermark(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    let path = require_file(state)?;
    let (positional, flags) = parse_repl_args(args);

    let text = positional.first().ok_or_else(|| {
        pdf_oxide::Error::InvalidOperation(
            "Usage: watermark <text> [-o out.pdf] [--pages 1-3]".to_string(),
        )
    })?;

    let opacity: f32 = flags
        .get("opacity")
        .map(|s| s.parse().unwrap_or(0.3))
        .unwrap_or(0.3);
    let rotation: f32 = flags
        .get("rotation")
        .map(|s| s.parse().unwrap_or(45.0))
        .unwrap_or(45.0);
    let font_size: f32 = flags
        .get("font-size")
        .map(|s| s.parse().unwrap_or(48.0))
        .unwrap_or(48.0);
    let color = flags.get("color").map(|s| s.as_str());
    let pages = flags.get("pages").map(|s| s.as_str());
    let output = flags.get("o").map(PathBuf::from);

    super::commands::watermark::run(
        &path,
        text,
        opacity,
        rotation,
        font_size,
        color,
        pages,
        output.as_deref(),
        state.password.as_deref(),
    )
}

fn cmd_flatten(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    let path = require_file(state)?;
    let (_positional, flags) = parse_repl_args(args);

    let forms = flags.contains_key("forms");
    let annotations = flags.contains_key("annotations");
    let output = flags.get("o").map(PathBuf::from);

    super::commands::flatten::run(
        &path,
        forms,
        annotations,
        output.as_deref(),
        state.password.as_deref(),
    )
}

fn cmd_crop(state: &mut ReplState, args: &str) -> pdf_oxide::Result<()> {
    let path = require_file(state)?;
    let (positional, flags) = parse_repl_args(args);

    let margins = positional.first().ok_or_else(|| {
        pdf_oxide::Error::InvalidOperation(
            "Usage: crop <l,r,t,b> [-o out.pdf] [--pages 1-3]".to_string(),
        )
    })?;

    let pages = flags.get("pages").map(|s| s.as_str());
    let output = flags.get("o").map(PathBuf::from);

    super::commands::crop::run(&path, margins, pages, output.as_deref(), state.password.as_deref())
}
