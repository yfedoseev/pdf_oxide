//! Struct-tree-scope `/ActualText` end-to-end tests (ISO 32000-1:2008
//! §14.9.4).
//!
//! Each fixture is a tiny hand-assembled tagged PDF that exercises one
//! aspect of the ActualText pipeline. The fixtures are built at test
//! time via [`PdfBuilder`] so they stay byte-stable and self-contained
//! (no Python at test time, no committed binary blobs).
//!
//! The pipeline is verified at four surfaces:
//!   - `extract_text`
//!   - `to_markdown`
//!   - `to_html`
//!   - `extract_structured`
//!
//! Tests pin the locked-in design decisions:
//!   - Inner ActualText wins over outer.
//!   - MC-scope ActualText wins over an enclosing struct-tree
//!     ActualText for the same MCID (the MC-scope replacement is
//!     applied at the content-stream layer; the struct-tree path
//!     records the MCID as a leaf with the inner text already
//!     baked in).
//!   - Multi-MCID subtrees emit the replacement ONCE.
//!   - Multi-page subtrees emit ONCE on the first page.
//!   - `/Alt` is supplemental, not a replacement.
//!   - Figure works the same as Span.
//!   - Unicode (CJK + RTL + emoji) round-trips.
//!   - `/MarkInfo /Suspects true` does NOT disable ActualText
//!     (decoupled per the design).
//!   - All four extraction surfaces agree.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;

// =============================================================
// Minimal hand-assembled tagged PDF builder.
//
// The builder lets each test describe a small set of structure
// elements + a content stream and produces a PDF byte slice that
// `PdfDocument::from_bytes` accepts. We intentionally keep it under
// 200 LoC and only support what these tests need (single page,
// Helvetica font, simple struct tree, ASCII content stream).
// =============================================================

/// A child of a structure element in the test DSL.
enum K {
    /// Marked content reference to `(mcid, page_index)`. Becomes an
    /// inline /MCR dict so the parser resolves it to a struct child
    /// whose page is the page object referenced by `page_index`.
    Mcid(u32, u32),
    /// Nested structure element (object number).
    Obj(u32),
}

/// One structure element for the test DSL.
struct Elem {
    obj_num: u32,
    s: &'static str,
    parent_obj: u32,
    page_obj: Option<u32>,
    actual_text: Option<String>,
    alt: Option<String>,
    children: Vec<K>,
}

impl Elem {
    fn new(obj_num: u32, s: &'static str, parent_obj: u32) -> Self {
        Self {
            obj_num,
            s,
            parent_obj,
            page_obj: None,
            actual_text: None,
            alt: None,
            children: Vec::new(),
        }
    }
    fn page(mut self, page_obj: u32) -> Self {
        self.page_obj = Some(page_obj);
        self
    }
    fn actual_text(mut self, t: &str) -> Self {
        self.actual_text = Some(t.to_string());
        self
    }
    fn alt(mut self, t: &str) -> Self {
        self.alt = Some(t.to_string());
        self
    }
    fn k(mut self, child: K) -> Self {
        self.children.push(child);
        self
    }
}

struct PdfBuilder {
    /// Content streams, one per page (each page is laid out with
    /// 0,0 at lower-left, 612x792 media box, single Helvetica).
    page_contents: Vec<Vec<u8>>,
    /// Structure elements. The first elem is the root (referenced by
    /// /StructTreeRoot /K).
    elems: Vec<Elem>,
    /// MCID -> StructElem object number, per page index. Used to
    /// build the ParentTree.
    parent_tree_entries: Vec<Vec<(u32, u32)>>,
    /// Whether `/MarkInfo /Suspects true` should be written.
    suspects: bool,
}

impl PdfBuilder {
    fn new() -> Self {
        Self {
            page_contents: Vec::new(),
            elems: Vec::new(),
            parent_tree_entries: Vec::new(),
            suspects: false,
        }
    }
    fn suspects(mut self) -> Self {
        self.suspects = true;
        self
    }
    fn add_page_content(&mut self, content: Vec<u8>) -> usize {
        self.page_contents.push(content);
        self.parent_tree_entries.push(Vec::new());
        self.page_contents.len() - 1
    }
    fn add_elem(&mut self, e: Elem) -> u32 {
        let n = e.obj_num;
        self.elems.push(e);
        n
    }
    fn register_mcid(&mut self, page_idx: usize, mcid: u32, struct_elem_obj: u32) {
        self.parent_tree_entries[page_idx].push((mcid, struct_elem_obj));
    }

    /// UTF-16BE hex-encoded PDF string, e.g. `<FEFF0066006C>`.
    fn pdf_string_utf16be(s: &str) -> String {
        let mut out = String::from("<FEFF");
        for u in s.encode_utf16() {
            out.push_str(&format!("{:04X}", u));
        }
        out.push('>');
        out
    }

    /// Build a PDF with an OCG layer whose `/Name` is `layer_name`.
    /// The content stream is expected to wrap its MCID-emitting BDC
    /// inside a parent `/OC /Properties << /OCG <ref> >>` BDC so the
    /// layer-exclusion path can suppress its glyphs. The layer is
    /// referenced from the page's `/Resources /Properties` dict under
    /// the name "L".
    fn build_with_ocg(self, layer_name: &str) -> Vec<u8> {
        // The OCG object number is appended after all the other
        // objects. We embed it via a string substitution shim on the
        // built bytes: simplest path is to pre-thread it into the
        // build pipeline. See `build_impl`.
        self.build_impl(Some(layer_name.to_string()))
    }

    fn build(self) -> Vec<u8> {
        self.build_impl(None)
    }

    fn build_impl(self, ocg_layer: Option<String>) -> Vec<u8> {
        use std::collections::BTreeMap;

        // Object numbering convention:
        //   1 = Catalog
        //   2 = Pages
        //   3 = Font (Helvetica)
        //   4..4+N-1 = Page dicts (one per page)
        //   4+N..4+2N-1 = Content streams
        //   4+2N = ParentTree
        //   4+2N+1 = StructTreeRoot
        //   4+2N+2.. = StructElems (assigned to start at builder obj nums)
        let n_pages = self.page_contents.len() as u32;

        // Decide reserved object numbers.
        let catalog = 1u32;
        let pages = 2u32;
        let font = 3u32;
        let first_page = 4u32;
        let first_content = first_page + n_pages;
        let parent_tree = first_content + n_pages;
        let struct_tree_root = parent_tree + 1;
        // OCG (optional content group) object, when used.
        let ocg = if ocg_layer.is_some() {
            Some(struct_tree_root + 1)
        } else {
            None
        };
        // Caller-side elem.obj_num MUST be >= first_struct_elem.
        let first_struct_elem = struct_tree_root + 1 + if ocg.is_some() { 1 } else { 0 };

        // Verify obj_nums are unique and high enough.
        let mut used = std::collections::HashSet::new();
        for e in &self.elems {
            assert!(
                e.obj_num >= first_struct_elem,
                "elem obj_num {} must be >= {}",
                e.obj_num,
                first_struct_elem
            );
            assert!(used.insert(e.obj_num), "duplicate elem obj_num {}", e.obj_num);
        }

        // The root struct elem is the first one in self.elems.
        let root_elem = self.elems[0].obj_num;

        // Collect objects keyed by number.
        let mut objs: BTreeMap<u32, Vec<u8>> = BTreeMap::new();

        // Catalog
        let mark_info = if self.suspects {
            "<< /Marked true /Suspects true >>"
        } else {
            "<< /Marked true >>"
        };
        let oc_props = match ocg {
            Some(o) => {
                format!(" /OCProperties << /OCGs [{} 0 R] /D << /Order [{} 0 R] >> >>", o, o)
            },
            None => String::new(),
        };
        objs.insert(
            catalog,
            format!(
                "<< /Type /Catalog /Pages {} 0 R /MarkInfo {} /StructTreeRoot {} 0 R{} >>",
                pages, mark_info, struct_tree_root, oc_props
            )
            .into_bytes(),
        );
        if let (Some(ocg_num), Some(layer_name)) = (ocg, ocg_layer.clone()) {
            objs.insert(ocg_num, format!("<< /Type /OCG /Name ({}) >>", layer_name).into_bytes());
        }

        // Pages tree.
        let kids: String = (0..n_pages)
            .map(|i| format!("{} 0 R", first_page + i))
            .collect::<Vec<_>>()
            .join(" ");
        objs.insert(
            pages,
            format!("<< /Type /Pages /Kids [{}] /Count {} >>", kids, n_pages).into_bytes(),
        );

        // Font.
        objs.insert(font, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec());

        // Pages + contents.
        for i in 0..n_pages {
            let content_obj = first_content + i;
            let content_bytes = &self.page_contents[i as usize];
            let stream_obj = format!("<< /Length {} >>\nstream\n", content_bytes.len());
            let mut stream = stream_obj.into_bytes();
            stream.extend_from_slice(content_bytes);
            stream.extend_from_slice(b"\nendstream");
            objs.insert(content_obj, stream);
            let props_entry = match ocg {
                Some(o) => format!(" /Properties << /L {} 0 R >>", o),
                None => String::new(),
            };
            objs.insert(
                first_page + i,
                format!(
                    "<< /Type /Page /Parent {} 0 R /MediaBox [0 0 612 792] \
                     /Resources << /Font << /F1 {} 0 R >> /ProcSet [/PDF /Text]{} >> \
                     /Contents {} 0 R /StructParents {} >>",
                    pages, font, props_entry, content_obj, i
                )
                .into_bytes(),
            );
        }

        // ParentTree: /Nums [0 [refs...] 1 [refs...] ...] one array per
        // page index, indexed by MCID.
        let mut nums = String::from("<< /Nums [");
        for (i, entries) in self.parent_tree_entries.iter().enumerate() {
            nums.push_str(&format!("{} [", i));
            // Sort by MCID, fill gaps with `null`.
            let mut by_mcid: Vec<(u32, u32)> = entries.clone();
            by_mcid.sort_by_key(|(m, _)| *m);
            let max_mcid = by_mcid.iter().map(|(m, _)| *m).max().unwrap_or(0);
            for m in 0..=max_mcid {
                if let Some((_, e)) = by_mcid.iter().find(|(mm, _)| *mm == m) {
                    nums.push_str(&format!("{} 0 R ", e));
                } else {
                    nums.push_str("null ");
                }
            }
            nums.push(']');
            if i + 1 < self.parent_tree_entries.len() {
                nums.push(' ');
            }
        }
        nums.push_str("] >>");
        objs.insert(parent_tree, nums.into_bytes());

        // StructTreeRoot.
        objs.insert(
            struct_tree_root,
            format!(
                "<< /Type /StructTreeRoot /K {} 0 R /ParentTree {} 0 R >>",
                root_elem, parent_tree
            )
            .into_bytes(),
        );

        // StructElems.
        for e in &self.elems {
            let mut body = format!("<< /Type /StructElem /S /{} /P {} 0 R", e.s, e.parent_obj);
            if let Some(pg) = e.page_obj {
                body.push_str(&format!(" /Pg {} 0 R", pg));
            }
            if let Some(ref at) = e.actual_text {
                body.push_str(&format!(" /ActualText {}", Self::pdf_string_utf16be(at)));
            }
            if let Some(ref alt) = e.alt {
                body.push_str(&format!(" /Alt {}", Self::pdf_string_utf16be(alt)));
            }
            // /K can be a single MCID integer, a single ref, or an array.
            // Use the MCR-dict form for MCID children so the parser
            // resolves /Pg from the dict and the descendant's page is
            // correct even when the bearing element has no /Pg of its
            // own (e.g. a multi-page heading).
            let mcr = |m: u32, pg_idx: u32| -> String {
                format!("<< /Type /MCR /Pg {} 0 R /MCID {} >>", first_page + pg_idx, m)
            };
            if e.children.len() == 1 {
                match &e.children[0] {
                    K::Mcid(m, p) => body.push_str(&format!(" /K {}", mcr(*m, *p))),
                    K::Obj(o) => body.push_str(&format!(" /K {} 0 R", o)),
                }
            } else if !e.children.is_empty() {
                body.push_str(" /K [");
                for (i, c) in e.children.iter().enumerate() {
                    if i > 0 {
                        body.push(' ');
                    }
                    match c {
                        K::Mcid(m, p) => body.push_str(&mcr(*m, *p)),
                        K::Obj(o) => body.push_str(&format!("{} 0 R", o)),
                    }
                }
                body.push(']');
            }
            body.push_str(" >>");
            objs.insert(e.obj_num, body.into_bytes());
        }

        // Write the PDF.
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n");
        let mut offsets: BTreeMap<u32, usize> = BTreeMap::new();
        for (num, body) in &objs {
            offsets.insert(*num, out.len());
            out.extend_from_slice(format!("{} 0 obj\n", num).as_bytes());
            out.extend_from_slice(body);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref_offset = out.len();
        let max_num = *objs.keys().max().unwrap();
        let n_objs = max_num + 1;
        out.extend_from_slice(format!("xref\n0 {}\n", n_objs).as_bytes());
        out.extend_from_slice(b"0000000000 65535 f \n");
        for n in 1..n_objs {
            if let Some(off) = offsets.get(&n) {
                out.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
            } else {
                out.extend_from_slice(b"0000000000 00000 f \n");
            }
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
                n_objs, xref_offset
            )
            .as_bytes(),
        );
        out
    }
}

/// One-MCID page content: a single /Span BDC ... EMC sequence with a
/// single Tj. Returns the content stream bytes.
fn single_mcid_content(prefix_text: &str, mcid: u32, raw_glyph_text: &str) -> Vec<u8> {
    format!(
        "BT\n/F1 12 Tf\n50 700 Td\n({}) Tj\n/Span << /MCID {} >> BDC\n({}) Tj\nEMC\nET\n",
        prefix_text, mcid, raw_glyph_text
    )
    .into_bytes()
}

/// Three-MCID page content for multi-MCID subtree tests: three /Span
/// BDC ... EMC sequences emitting three single-glyph runs.
fn three_mcid_content() -> Vec<u8> {
    let mut s = String::from("BT\n/F1 12 Tf\n50 700 Td\n");
    for m in [7u32, 8, 9] {
        s.push_str(&format!("/Span << /MCID {} >> BDC\n(X) Tj\nEMC\n", m));
    }
    s.push_str("ET\n");
    s.into_bytes()
}

/// Two-page page-0 content (one MCID) used by multi-page tests.
fn one_mcid_simple(mcid: u32) -> Vec<u8> {
    format!("BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID {} >> BDC\n(X) Tj\nEMC\nET\n", mcid)
        .into_bytes()
}

// ----------------------------------------------------------------------
// Fixtures and tests
// ----------------------------------------------------------------------

/// Fixture 1: simple single-Span /ActualText.
fn fixture_simple() -> Vec<u8> {
    // For 1 page: catalog=1, pages=2, font=3, page=4, content=5,
    // parent_tree=6, struct_tree_root=7, first elem obj = 8.
    let mut b = PdfBuilder::new();
    b.add_page_content(
        "BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(X) Tj\nEMC\nET\n"
            .as_bytes()
            .to_vec(),
    );
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("fi")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    b.build()
}

#[test]
fn actualtext_simple_emits_replacement_no_raw() {
    let pdf = fixture_simple();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let opts = ConversionOptions::default();

    let extracted = doc.extract_text(0).expect("extract_text");
    let md = doc.to_markdown(0, &opts).expect("to_markdown");
    let html = doc.to_html(0, &opts).expect("to_html");
    let plain = doc.to_plain_text(0, &opts).expect("to_plain_text");

    for (name, output) in [
        ("extract_text", &extracted),
        ("to_markdown", &md),
        ("to_html", &html),
        ("to_plain_text", &plain),
    ] {
        assert!(output.contains("fi"), "{}: expected 'fi' (replacement), got {:?}", name, output);
        assert!(!output.contains('X'), "{}: raw 'X' must NOT appear, got {:?}", name, output);
    }
}

/// Fixture 2: nested outer/inner — inner wins.
fn fixture_nested_inner_wins() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    b.add_page_content(
        "BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(X) Tj\nEMC\nET\n"
            .as_bytes()
            .to_vec(),
    );
    // Outer Span (obj 8) wraps Inner Span (obj 9). Root elem comes first.
    let _outer = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("outer")
            .k(K::Obj(9)),
    );
    let _inner = b.add_elem(
        Elem::new(9, "Span", 8)
            .page(4)
            .actual_text("inner")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 9);
    b.build()
}

#[test]
fn actualtext_nested_inner_wins() {
    let pdf = fixture_nested_inner_wins();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let opts = ConversionOptions::default();

    let extracted = doc.extract_text(0).expect("extract_text");
    let md = doc.to_markdown(0, &opts).expect("to_markdown");
    let html = doc.to_html(0, &opts).expect("to_html");
    let plain = doc.to_plain_text(0, &opts).expect("to_plain_text");

    for (name, output) in [
        ("extract_text", &extracted),
        ("to_markdown", &md),
        ("to_html", &html),
        ("to_plain_text", &plain),
    ] {
        assert!(
            output.contains("inner"),
            "{}: inner replacement must appear, got {:?}",
            name,
            output
        );
        assert!(
            !output.contains("outer"),
            "{}: outer replacement must NOT appear (inner-wins), got {:?}",
            name,
            output
        );
        assert!(!output.contains('X'), "{}: raw 'X' must NOT appear, got {:?}", name, output);
    }
}

/// Fixture 4: multi-MCID subtree — one Span /ActualText 'expanded' /K [7 8 9].
fn fixture_multi_mcid_subtree() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    b.add_page_content(three_mcid_content());
    let _span = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("expanded")
            .k(K::Mcid(7, 0))
            .k(K::Mcid(8, 0))
            .k(K::Mcid(9, 0)),
    );
    b.register_mcid(0, 7, 8);
    b.register_mcid(0, 8, 8);
    b.register_mcid(0, 9, 8);
    b.build()
}

#[test]
fn actualtext_multi_mcid_subtree_emits_once() {
    let pdf = fixture_multi_mcid_subtree();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let opts = ConversionOptions::default();

    let extracted = doc.extract_text(0).expect("extract_text");
    let md = doc.to_markdown(0, &opts).expect("to_markdown");
    let html = doc.to_html(0, &opts).expect("to_html");

    for (name, output) in [
        ("extract_text", &extracted),
        ("to_markdown", &md),
        ("to_html", &html),
    ] {
        assert_eq!(
            output.matches("expanded").count(),
            1,
            "{}: expected 'expanded' exactly once, got {:?}",
            name,
            output
        );
        assert!(!output.contains('X'), "{}: raw 'X' must NOT appear, got {:?}", name, output);
    }
}

/// Fixture 5: multi-page subtree — /H1 covers MCIDs on pages 0 and 1.
fn fixture_multi_page() -> Vec<u8> {
    // For 2 pages: catalog=1, pages=2, font=3, page0=4, page1=5,
    // content0=6, content1=7, parent_tree=8, struct_tree_root=9,
    // first elem obj = 10.
    let mut b = PdfBuilder::new();
    b.add_page_content(one_mcid_simple(0));
    b.add_page_content(one_mcid_simple(0));
    let h1 = b.add_elem(
        Elem::new(10, "H1", 9)
            .actual_text("Heading X")
            // Two children, one per page.
            .k(K::Mcid(0, 0))
            .k(K::Mcid(0, 1)),
    );
    b.register_mcid(0, 0, h1);
    b.register_mcid(1, 0, h1);
    b.build()
}

#[test]
fn actualtext_multi_page_emits_only_first_page() {
    let pdf = fixture_multi_page();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let opts = ConversionOptions::default();

    // Page 0 must contain the replacement.
    let p0_text = doc.extract_text(0).expect("extract_text 0");
    let p0_md = doc.to_markdown(0, &opts).expect("md 0");
    assert!(
        p0_text.contains("Heading X"),
        "page 0 extract_text must contain replacement, got {:?}",
        p0_text
    );
    assert!(
        p0_md.contains("Heading X"),
        "page 0 markdown must contain replacement, got {:?}",
        p0_md
    );

    // Page 1 must NOT contain the replacement.
    let p1_text = doc.extract_text(1).expect("extract_text 1");
    let p1_md = doc.to_markdown(1, &opts).expect("md 1");
    assert!(
        !p1_text.contains("Heading X"),
        "page 1 extract_text must NOT repeat the replacement, got {:?}",
        p1_text
    );
    assert!(
        !p1_md.contains("Heading X"),
        "page 1 markdown must NOT repeat the replacement, got {:?}",
        p1_md
    );
    // And page 1 must NOT leak the raw glyph either.
    assert!(!p1_text.contains('X'), "page 1 raw glyph leak, got {:?}", p1_text);
}

/// Fixture 6: /Alt + /ActualText — ActualText wins, /Alt is not output.
fn fixture_with_alt() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    b.add_page_content(single_mcid_content("", 0, "X"));
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("real")
            .alt("fallback alt description")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    b.build()
}

#[test]
fn actualtext_wins_over_alt() {
    let pdf = fixture_with_alt();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let opts = ConversionOptions::default();
    let extracted = doc.extract_text(0).expect("extract_text");
    let md = doc.to_markdown(0, &opts).expect("md");
    for (name, output) in [("extract_text", &extracted), ("to_markdown", &md)] {
        assert!(
            output.contains("real"),
            "{}: ActualText 'real' must appear, got {:?}",
            name,
            output
        );
        assert!(
            !output.contains("fallback"),
            "{}: /Alt must NOT appear in extraction output, got {:?}",
            name,
            output
        );
    }
}

/// Fixture 7: only /Alt, no /ActualText — raw glyph preserved.
fn fixture_only_alt() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    b.add_page_content(single_mcid_content("", 0, "X"));
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .alt("alt only")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    b.build()
}

#[test]
fn alt_alone_does_not_replace_raw_glyph() {
    let pdf = fixture_only_alt();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    assert!(
        extracted.contains('X'),
        "raw 'X' must appear when only /Alt is present, got {:?}",
        extracted
    );
}

/// Fixture 8: /Figure /ActualText 'logo text'.
fn fixture_figure() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    b.add_page_content(single_mcid_content("", 0, "X"));
    let _ = b.add_elem(
        Elem::new(8, "Figure", 7)
            .page(4)
            .actual_text("logo text")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    b.build()
}

#[test]
fn actualtext_on_figure_emits_replacement() {
    let pdf = fixture_figure();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    assert!(
        extracted.contains("logo text"),
        "figure ActualText must appear, got {:?}",
        extracted
    );
    assert!(!extracted.contains('X'), "raw 'X' must NOT appear, got {:?}", extracted);
}

/// Fixture 9: Unicode (CJK + RTL + emoji) ActualText.
fn fixture_unicode() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    b.add_page_content(single_mcid_content("", 0, "X"));
    // 你好 (Chinese), שלום (Hebrew RTL), 🚀 (emoji)
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("你好 שלום 🚀")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    b.build()
}

#[test]
fn actualtext_unicode_round_trips() {
    let pdf = fixture_unicode();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    assert!(extracted.contains("你好"), "CJK characters must appear, got {:?}", extracted);
    assert!(
        extracted.contains("שלום"),
        "RTL Hebrew characters must appear, got {:?}",
        extracted
    );
    assert!(extracted.contains("🚀"), "emoji must appear, got {:?}", extracted);
}

/// Fixture 10: /MarkInfo /Suspects true — ActualText still emits.
fn fixture_suspects_true() -> Vec<u8> {
    let mut b = PdfBuilder::new().suspects();
    b.add_page_content(single_mcid_content("", 0, "X"));
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("replacement")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    b.build()
}

#[test]
fn actualtext_emits_when_suspects_true() {
    let pdf = fixture_suspects_true();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    assert!(
        extracted.contains("replacement"),
        "ActualText must emit even when /MarkInfo /Suspects is true (decoupled from reading-order trust), got {:?}",
        extracted
    );
    assert!(!extracted.contains('X'), "raw 'X' must NOT appear, got {:?}", extracted);
}

/// Fixture 3: MC-scope ActualText nested inside a struct-tree
/// ActualText scope. PDF spec precedence: MC-scope (inner, content
/// stream) wins over struct-tree (outer) for the same MCID, because
/// the in-stream replacement is the most specific declaration.
fn fixture_descendant_mc_scope() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    // BDC carries inline /ActualText "mc" — content stream rewrites
    // raw glyph "X" to "mc" at extraction time.
    let content = "BT\n/F1 12 Tf\n50 700 Td\n\
                   /Span << /MCID 0 /ActualText <FEFF006D0063> >> BDC\n\
                   (X) Tj\n\
                   EMC\nET\n";
    b.add_page_content(content.as_bytes().to_vec());
    // StructElem on the same MCID carries /ActualText "struct".
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("struct")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    b.build()
}

#[test]
fn mc_scope_actualtext_wins_over_struct_scope() {
    let pdf = fixture_descendant_mc_scope();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    // Precedence: MC-scope replacement is innermost and most
    // specific; it must beat the enclosing struct-tree /ActualText.
    assert!(
        extracted.contains("mc"),
        "MC-scope replacement 'mc' must appear (innermost wins), got {:?}",
        extracted
    );
    assert!(
        !extracted.contains("struct"),
        "struct-tree replacement 'struct' must NOT appear (MC-scope wins), got {:?}",
        extracted
    );
    assert!(!extracted.contains('X'), "raw 'X' must NOT appear, got {:?}", extracted);
}

/// Fixture 11a: All covered MCIDs are inside an excluded OCG layer —
/// emission must be skipped.
fn fixture_ocg_excluded_all() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    // Wrap the MCID's BDC inside an OCG /OC /L scope (L is the name
    // declared in /Resources /Properties).
    let content = "BT\n/F1 12 Tf\n50 700 Td\n\
                   /OC /L BDC\n\
                   /Span << /MCID 0 >> BDC\n\
                   (X) Tj\n\
                   EMC\n\
                   EMC\nET\n";
    b.add_page_content(content.as_bytes().to_vec());
    // Object num accounting: when an OCG is added, first_struct_elem
    // shifts up by 1 (see build_impl).
    let _ = b.add_elem(
        Elem::new(9, "Span", 7)
            .page(4)
            .actual_text("replacement")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 9);
    b.build_with_ocg("HiddenLayer")
}

#[test]
fn actualtext_ocg_excluded_all_suppresses_emission() {
    let pdf = fixture_ocg_excluded_all();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let mut excluded = std::collections::HashSet::new();
    excluded.insert("HiddenLayer".to_string());
    let extracted = doc
        .extract_text_filtered(0, excluded, std::collections::HashSet::new())
        .expect("extract_text_filtered");
    assert!(
        !extracted.contains("replacement"),
        "ActualText emission must be skipped when every covered MCID is OCG-excluded, got {:?}",
        extracted
    );
    assert!(
        !extracted.contains('X'),
        "raw 'X' must also be suppressed under OCG exclusion, got {:?}",
        extracted
    );
}

#[test]
fn actualtext_ocg_visible_path_still_emits() {
    // Counter-test: when the OCG layer is NOT excluded, the
    // emission must still fire (proves the suppression is layer-
    // gated, not always-on).
    let pdf = fixture_ocg_excluded_all();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    assert!(
        extracted.contains("replacement"),
        "without excluding the OCG, the emission must fire normally, got {:?}",
        extracted
    );
}

/// Fixture 12: cross-path consistency — extract_text, to_markdown,
/// to_html, to_plain_text, and extract_structured all see the
/// replacement text and not the raw glyph.
#[test]
fn actualtext_cross_path_consistency() {
    let pdf = fixture_simple();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let opts = ConversionOptions::default();
    let surfaces: Vec<(&str, String)> = vec![
        ("extract_text", doc.extract_text(0).expect("extract_text")),
        ("to_markdown", doc.to_markdown(0, &opts).expect("md")),
        ("to_html", doc.to_html(0, &opts).expect("html")),
        ("to_plain_text", doc.to_plain_text(0, &opts).expect("plain")),
        (
            "extract_structured",
            doc.extract_structured(0)
                .expect("structured")
                .regions
                .iter()
                .map(|r| r.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        ),
    ];
    for (name, out) in &surfaces {
        assert!(out.contains("fi"), "{}: expected 'fi' on every surface, got {:?}", name, out);
        assert!(
            !out.contains('X'),
            "{}: raw 'X' must NOT appear on any surface, got {:?}",
            name,
            out
        );
    }
}
