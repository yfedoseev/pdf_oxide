//! Regression tests for issue #509:
//! "Rejects Linearized PDF when %PDF- header is offset from byte 0".
//!
//! Two independent defects, either of which sank the kreuzberg `medium.pdf`
//! corpus file (a Linearized PDF whose `%PDF-` header is preceded by ~364
//! bytes of captive-portal HTML, and whose sparse final trailer legitimately
//! omits `/Root`):
//!
//!   1. The garbage-prefix xref-offset shift was gated on the final trailer
//!      carrying `/Root` — so a `/Root`-less trailer skipped the shift.
//!   2. xref reconstruction accepted a parsed-but-`/Root`-less trailer
//!      instead of falling through to Catalog discovery, and `catalog()`
//!      had no fallback when the trailer omitted `/Root`.
//!
//! These build inline PDFs whose xref offsets are *logical* (relative to the
//! `%PDF-` position), then optionally prepend leading garbage — exactly the
//! shape of the real file — so no 245 KB external fixture is needed.

use pdf_oxide::document::PdfDocument;

/// Build a minimal single-page PDF whose internal byte offsets are all
/// relative to the `%PDF-` header (logical byte 0). `with_root` controls
/// whether the final trailer dict carries `/Root` (a Linearized file's
/// sparse end-of-file trailer legitimately omits it).
fn build_logical_pdf(with_root: bool) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> >>\nendobj\n",
    );

    let xref_off = pdf.len();
    pdf.extend_from_slice(b"xref\n0 4\n");
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off1).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off2).as_bytes());
    pdf.extend_from_slice(format!("{:010} 00000 n \n", off3).as_bytes());

    let trailer = if with_root {
        format!("trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref_off}\n%%EOF\n")
    } else {
        // Sparse Linearized-style trailer: NO /Root.
        format!("trailer\n<< /Size 4 >>\nstartxref\n{xref_off}\n%%EOF\n")
    };
    pdf.extend_from_slice(trailer.as_bytes());
    pdf
}

/// ~380 bytes of injected captive-portal HTML, mirroring the real file's
/// `dnserrorassist.att.net` redirect blob. Long enough that the xref
/// keyword-tolerance scan cannot bridge it.
fn html_redirect_garbage() -> Vec<u8> {
    let mut g = Vec::new();
    g.extend_from_slice(
        b"<html><head><meta http-equiv=\"refresh\" content=\"0;url=http://dnserrorassist.example.net/redirect?\
          orig=http://example.org/medium.pdf\"></head><body>\
          <script type=\"text/javascript\">window.location=\"http://dnserrorassist.example.net/redirect\";</script>\
          <p>Your DNS request has been redirected by your network provider. \
          If you are not redirected automatically, follow the link.</p>\
          </body></html>\n",
    );
    g
}

/// The full #509 scenario: leading garbage AND a sparse final trailer that
/// omits `/Root`. Before the fix this failed with
/// "Invalid PDF: Trailer missing /Root entry".
#[test]
fn test_issue_509_garbage_prefix_sparse_trailer() {
    let mut bytes = html_redirect_garbage();
    let offset = bytes.len();
    bytes.extend_from_slice(&build_logical_pdf(false));

    let doc = PdfDocument::from_bytes(bytes)
        .unwrap_or_else(|e| panic!("from_bytes failed for garbage-prefixed PDF: {e}"));

    let (major, minor) = doc.version();
    assert_eq!((major, minor), (1, 4), "header parsed from offset {offset}");

    let pages = doc
        .page_count()
        .unwrap_or_else(|e| panic!("page_count failed (issue #509 regression): {e}"));
    assert_eq!(pages, 1, "single-page PDF must report 1 page");
}

/// A well-formed PDF (no leading garbage) whose final trailer omits
/// `/Root`. NOTE: this exercises the *xref-reconstruction* Catalog
/// synthesis path, not `catalog()`'s `find_catalog_by_scan` fallback —
/// the public open path fails root validation on the `/Root`-less trailer
/// and reconstructs a `/Root`-bearing trailer before `catalog()` runs
/// (the `catalog()` fallback itself is covered by the unit test
/// `document::tests::test_catalog_recovers_when_trailer_omits_root`).
/// Either way the Catalog is discovered by scanning objects for
/// `/Type /Catalog`, which is what Poppler / PDFium do.
#[test]
fn test_issue_509_catalog_fallback_when_trailer_omits_root() {
    let doc = PdfDocument::from_bytes(build_logical_pdf(false))
        .expect("from_bytes failed for /Root-less trailer PDF");

    let pages = doc
        .page_count()
        .expect("page_count must recover via /Type /Catalog scan when trailer omits /Root");
    assert_eq!(pages, 1);
}

/// Regression guard: a garbage-prefixed PDF whose trailer DOES carry `/Root`
/// must keep working (the existing header-offset path must not regress).
#[test]
fn test_issue_509_garbage_prefix_with_root_still_works() {
    let mut bytes = html_redirect_garbage();
    bytes.extend_from_slice(&build_logical_pdf(true));

    let doc = PdfDocument::from_bytes(bytes).expect("garbage-prefixed /Root PDF must still load");
    assert_eq!(doc.page_count().expect("page_count"), 1);
}

/// Build a PDF with the **real Linearized shape**: an earlier `trailer`
/// that carries `/Root` (the first-page xref chain) followed by a later,
/// sparse end-of-file `trailer` that omits `/Root` (only `/Size` + `/ID`,
/// exactly like the issue #509 `medium.pdf`: `<</Size 114/ID[...]>>`).
fn build_two_trailer_linearized() -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> >>\nendobj\n",
    );

    let xref_off = pdf.len();
    pdf.extend_from_slice(b"xref\n0 4\n");
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    pdf.extend_from_slice(format!("{off1:010} 00000 n \n").as_bytes());
    pdf.extend_from_slice(format!("{off2:010} 00000 n \n").as_bytes());
    pdf.extend_from_slice(format!("{off3:010} 00000 n \n").as_bytes());

    // FIRST trailer — carries /Root (Linearized first-chain trailer).
    pdf.extend_from_slice(
        format!("trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref_off}\n%%EOF\n").as_bytes(),
    );

    // LATER, sparse end-of-file trailer — NO /Root, only /Size + /ID,
    // appearing *after* the /Root-bearing one in byte order. Pre-fix,
    // `find_trailer` kept the last parsed trailer regardless of /Root, so
    // this one clobbered the good one and the load failed with
    // "Trailer missing /Root entry".
    let xref2_off = pdf.len();
    pdf.extend_from_slice(b"xref\n0 1\n0000000000 65535 f \n");
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size 4 /ID[<AAAAAAAA><BBBBBBBB>] >>\nstartxref\n{xref2_off}\n%%EOF\n"
        )
        .as_bytes(),
    );
    pdf
}

/// The discriminating #509 case: two trailers, the *later* one sparse and
/// `/Root`-less. `find_trailer` must keep the earlier `/Root`-bearing
/// trailer, not the last-parsed one. This is the exact real-file shape;
/// the single-trailer tests above do NOT exercise the "skip a later
/// `/Root`-less trailer, keep the earlier `/Root`-bearing one" logic, so a
/// future revert of that logic would pass them while silently re-breaking
/// the real `medium.pdf`. Garbage prefix forces the reconstruction path
/// where `find_trailer` runs.
#[test]
fn test_issue_509_linearized_two_trailers_keeps_root_bearing() {
    let mut bytes = html_redirect_garbage();
    bytes.extend_from_slice(&build_two_trailer_linearized());

    let doc = PdfDocument::from_bytes(bytes)
        .unwrap_or_else(|e| panic!("two-trailer Linearized PDF must load (#509): {e}"));
    assert_eq!(
        doc.page_count().unwrap_or_else(|e| panic!(
            "page_count failed — later /Root-less trailer wrongly won: {e}"
        )),
        1
    );

    // Same shape without the garbage prefix: the final `startxref` points at
    // the second (sparse, valid) xref, so the regular xref parse *succeeds*,
    // but that final trailer has no /Root, so root validation fails and
    // reconstruction runs anyway. The earlier /Root-bearing trailer must
    // still win.
    let doc2 = PdfDocument::from_bytes(build_two_trailer_linearized())
        .expect("two-trailer Linearized PDF (no prefix) must load");
    assert_eq!(doc2.page_count().expect("page_count"), 1);
}
