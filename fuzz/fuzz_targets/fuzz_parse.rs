#![no_main]

use libfuzzer_sys::fuzz_target;
use pdf_oxide::PdfDocument;

// Feed arbitrary bytes to the in-memory parse entry point, then exercise a
// little more of the pipeline on anything that parses. The contract under test
// is robustness: no input may panic, overflow, or hang — malformed PDFs must
// surface as `Err`, not a crash.
fuzz_target!(|data: &[u8]| {
    if let Ok(doc) = PdfDocument::from_bytes(data.to_vec()) {
        // Only touch a real parse succeeded — extract text from the first page
        // to reach the content-stream / font / cmap paths as well.
        let _ = doc.extract_text(0);
    }
});
