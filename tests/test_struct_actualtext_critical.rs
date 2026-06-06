//! Second-pass regression tests for struct-tree-scope `/ActualText`
//! (ISO 32000-1:2008 §14.9.4) that cover the silent-content-loss bugs
//! the per-emission anchor model exhibited and the architectural
//! choices that fix them.
//!
//! Each test pins a specific spec-correct behaviour and reproduces a
//! concrete shape that previously dropped content or emitted the wrong
//! text. The fixtures are built on top of the same minimal
//! `PdfBuilder` used by `test_struct_actualtext.rs`; we duplicate the
//! relevant pieces here to keep the test crate self-contained while
//! avoiding cross-module visibility shenanigans.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;

// =============================================================
// Minimal hand-assembled tagged PDF builder. (Carbon copy of the
// helper from `test_struct_actualtext.rs`; integration tests are
// independent crates so we cannot import it.)
// =============================================================

enum K {
    Mcid(u32, u32),
    Obj(u32),
}

struct Elem {
    obj_num: u32,
    s: &'static str,
    parent_obj: u32,
    page_obj: Option<u32>,
    actual_text: Option<String>,
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
    fn k(mut self, child: K) -> Self {
        self.children.push(child);
        self
    }
}

struct PdfBuilder {
    page_contents: Vec<Vec<u8>>,
    elems: Vec<Elem>,
    parent_tree_entries: Vec<Vec<(u32, u32)>>,
}

impl PdfBuilder {
    fn new() -> Self {
        Self {
            page_contents: Vec::new(),
            elems: Vec::new(),
            parent_tree_entries: Vec::new(),
        }
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

    fn pdf_string_utf16be(s: &str) -> String {
        let mut out = String::from("<FEFF");
        for u in s.encode_utf16() {
            out.push_str(&format!("{:04X}", u));
        }
        out.push('>');
        out
    }

    fn build(self) -> Vec<u8> {
        self.build_impl(None)
    }

    fn build_with_ocg(self, layer_name: &str) -> Vec<u8> {
        self.build_impl(Some(layer_name.to_string()))
    }

    fn build_impl(self, ocg_layer: Option<String>) -> Vec<u8> {
        use std::collections::BTreeMap;
        let n_pages = self.page_contents.len() as u32;
        let catalog = 1u32;
        let pages = 2u32;
        let font = 3u32;
        let first_page = 4u32;
        let first_content = first_page + n_pages;
        let parent_tree = first_content + n_pages;
        let struct_tree_root = parent_tree + 1;
        let ocg = if ocg_layer.is_some() {
            Some(struct_tree_root + 1)
        } else {
            None
        };
        let first_struct_elem = struct_tree_root + 1 + if ocg.is_some() { 1 } else { 0 };

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
        let root_elem = self.elems[0].obj_num;

        let mut objs: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
        let oc_props = match ocg {
            Some(o) => {
                format!(" /OCProperties << /OCGs [{} 0 R] /D << /Order [{} 0 R] >> >>", o, o)
            },
            None => String::new(),
        };
        objs.insert(
            catalog,
            format!(
                "<< /Type /Catalog /Pages {} 0 R /MarkInfo << /Marked true >> /StructTreeRoot {} 0 R{} >>",
                pages, struct_tree_root, oc_props
            )
            .into_bytes(),
        );
        if let (Some(ocg_num), Some(layer_name)) = (ocg, ocg_layer.clone()) {
            objs.insert(ocg_num, format!("<< /Type /OCG /Name ({}) >>", layer_name).into_bytes());
        }
        let kids: String = (0..n_pages)
            .map(|i| format!("{} 0 R", first_page + i))
            .collect::<Vec<_>>()
            .join(" ");
        objs.insert(
            pages,
            format!("<< /Type /Pages /Kids [{}] /Count {} >>", kids, n_pages).into_bytes(),
        );
        objs.insert(font, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec());

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

        let mut nums = String::from("<< /Nums [");
        for (i, entries) in self.parent_tree_entries.iter().enumerate() {
            nums.push_str(&format!("{} [", i));
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

        objs.insert(
            struct_tree_root,
            format!(
                "<< /Type /StructTreeRoot /K {} 0 R /ParentTree {} 0 R >>",
                root_elem, parent_tree
            )
            .into_bytes(),
        );

        for e in &self.elems {
            let mut body = format!("<< /Type /StructElem /S /{} /P {} 0 R", e.s, e.parent_obj);
            if let Some(pg) = e.page_obj {
                body.push_str(&format!(" /Pg {} 0 R", pg));
            }
            if let Some(ref at) = e.actual_text {
                body.push_str(&format!(" /ActualText {}", Self::pdf_string_utf16be(at)));
            }
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

/// Two-MCID page content: two `/Span` BDC ... EMC sequences, each
/// emitting a single glyph. Used by tests that exercise structurally
/// distinct MCIDs side by side. Td is relative (§9.4.2) so the
/// second one moves the cursor down a line, keeping both spans on-page.
fn two_mcid_content(g0: char, g1: char) -> Vec<u8> {
    format!(
        "BT\n/F1 12 Tf\n50 700 Td\n\
         /Span << /MCID 0 >> BDC\n({}) Tj\nEMC\n\
         0 -20 Td\n\
         /Span << /MCID 1 >> BDC\n({}) Tj\nEMC\nET\n",
        g0, g1
    )
    .into_bytes()
}

// =============================================================
// CRITICAL-1: nested ActualText with sibling MCIDs under outer.
// =============================================================
//
// Shape:
//   Outer Span /ActualText "O"
//     /K [
//       Inner Span /ActualText "I"  /K MCID 0
//       MCID 1                                // direct child of OUTER
//     ]
//
// Expected: "I" for MCID 0 (inner-wins) and "O" for MCID 1 (outer covers it).
// Previous-pass bug: only "I" appeared; MCID 1's "O" silently vanished
// because the outer emission was "shadowed" by the inner whose
// covered_mcids was a strict subset of the outer's.

fn fixture_nested_with_outer_sibling() -> Vec<u8> {
    // n_pages=1 → catalog=1 pages=2 font=3 page=4 content=5 ptree=6 strtreeroot=7
    // first elem obj = 8.
    let mut b = PdfBuilder::new();
    b.add_page_content(two_mcid_content('X', 'Y'));
    let _outer = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("O")
            .k(K::Obj(9))
            .k(K::Mcid(1, 0)),
    );
    let _inner = b.add_elem(
        Elem::new(9, "Span", 8)
            .page(4)
            .actual_text("I")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 9); // MCID 0 belongs to the inner Span
    b.register_mcid(0, 1, 8); // MCID 1 belongs to the outer Span
    b.build()
}

#[test]
fn critical1_nested_with_outer_sibling_emits_both() {
    let pdf = fixture_nested_with_outer_sibling();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let opts = ConversionOptions::default();

    let surfaces: Vec<(&str, String)> = vec![
        ("extract_text", doc.extract_text(0).expect("extract_text")),
        ("to_markdown", doc.to_markdown(0, &opts).expect("md")),
        ("to_html", doc.to_html(0, &opts).expect("html")),
        ("to_plain_text", doc.to_plain_text(0, &opts).expect("plain")),
    ];

    for (name, out) in &surfaces {
        assert!(out.contains('I'), "{}: inner replacement 'I' must appear, got {:?}", name, out);
        assert!(
            out.contains('O'),
            "{}: outer replacement 'O' must appear for the sibling MCID, got {:?}",
            name,
            out
        );
        // Raw glyphs are suppressed (covered by some ActualText scope).
        assert!(!out.contains('X'), "{}: raw 'X' must NOT appear, got {:?}", name, out);
        assert!(!out.contains('Y'), "{}: raw 'Y' must NOT appear, got {:?}", name, out);
    }
}

// =============================================================
// CRITICAL-2: cross-page MCID collision.
// =============================================================
//
// Page 0 has H1 /ActualText "Heading" covering page-0 MCID 0.
// Page 1 has an unrelated /P /K MCID 0 with raw glyph "B".
// The two MCID 0s are independent (MCIDs are per-page per spec).
// Previous-pass bug: `covered_mcids: HashSet<u32>` collided across
// pages, so page-1's MCID 0 was treated as covered too and its "B"
// vanished.

fn fixture_cross_page_mcid_collision() -> Vec<u8> {
    // n_pages=2 → catalog=1 pages=2 font=3 page0=4 page1=5 content0=6
    // content1=7 ptree=8 strtreeroot=9 first elem obj = 10.
    let mut b = PdfBuilder::new();
    // Page 0: a single MCID 0 inside a /Span BDC emitting "A".
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n\
        /Span << /MCID 0 >> BDC\n(A) Tj\nEMC\nET\n"
            .to_vec(),
    );
    // Page 1: a single MCID 0 inside a /Span BDC emitting "B" — NOT covered.
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n\
        /Span << /MCID 0 >> BDC\n(B) Tj\nEMC\nET\n"
            .to_vec(),
    );

    // Page-0 H1 with /ActualText "Heading" covering page-0 MCID 0.
    let h1 = b.add_elem(
        Elem::new(10, "H1", 9)
            .actual_text("Heading")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, h1);

    // Page-1 plain /P (no /ActualText) wrapping page-1 MCID 0.
    let p = b.add_elem(Elem::new(11, "P", 9).k(K::Mcid(0, 1)));
    b.register_mcid(1, 0, p);
    b.build()
}

#[test]
fn critical2_cross_page_mcid_collision_does_not_swallow_unrelated_text() {
    let pdf = fixture_cross_page_mcid_collision();
    let doc = PdfDocument::from_bytes(pdf).expect("open");

    let p0 = doc.extract_text(0).expect("extract_text 0");
    let p1 = doc.extract_text(1).expect("extract_text 1");

    assert!(
        p0.contains("Heading"),
        "page 0 must emit the replacement 'Heading', got {:?}",
        p0
    );
    assert!(
        !p0.contains('A'),
        "page 0 raw glyph 'A' must be suppressed (covered), got {:?}",
        p0
    );

    assert!(
        p1.contains('B'),
        "page 1 raw glyph 'B' must NOT be silently dropped — its MCID is unrelated to page 0's covered MCID, got {:?}",
        p1
    );
    assert!(
        !p1.contains("Heading"),
        "page 1 must NOT see the page-0 replacement, got {:?}",
        p1
    );
}

// =============================================================
// CRITICAL-3: anchor MCID hidden by an excluded OCG layer; emission
// must still fire at a visible non-anchor MCID under the same scope.
// =============================================================
//
// Page content:
//   /OC /L BDC                  // hidden layer scope
//     /Span << /MCID 0 >> BDC
//       (X) Tj                  // anchor MCID — filtered out
//     EMC
//   EMC
//   /Span << /MCID 1 >> BDC
//     (Y) Tj                    // visible non-anchor under the same struct ActualText
//   EMC
//
// Structure:
//   Span /ActualText "R" /K [MCID 0, MCID 1]
//
// Expected: when the "L" layer is excluded, MCID 0's glyphs are
// hidden but MCID 1 stays visible — the emission must still produce
// "R" once (the replacement is for the union of MCIDs, and at least
// one is visible).

fn fixture_anchor_hidden_partial_ocg() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    // With an OCG, first_struct_elem shifts up by 1 → elem obj_num >= 9.
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
        /OC /L BDC\n\
          /Span << /MCID 0 >> BDC\n(X) Tj\nEMC\n\
        EMC\n\
        0 -20 Td\n\
        /Span << /MCID 1 >> BDC\n(Y) Tj\nEMC\nET\n";
    b.add_page_content(content.to_vec());

    let span = b.add_elem(
        Elem::new(9, "Span", 7)
            .page(4)
            .actual_text("R")
            .k(K::Mcid(0, 0))
            .k(K::Mcid(1, 0)),
    );
    b.register_mcid(0, 0, span);
    b.register_mcid(0, 1, span);
    b.build_with_ocg("HiddenLayer")
}

#[test]
fn critical3_partial_ocg_anchor_hidden_still_emits() {
    let pdf = fixture_anchor_hidden_partial_ocg();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let mut excluded = std::collections::HashSet::new();
    excluded.insert("HiddenLayer".to_string());
    let extracted = doc
        .extract_text_filtered(0, excluded, std::collections::HashSet::new())
        .expect("extract_text_filtered");

    assert!(
        extracted.contains('R'),
        "replacement 'R' must still emit when at least one covered MCID is visible, got {:?}",
        extracted
    );
    assert!(
        !extracted.contains('X'),
        "hidden glyph 'X' must NOT appear, got {:?}",
        extracted
    );
    assert!(
        !extracted.contains('Y'),
        "visible covered glyph 'Y' must be suppressed (covered by struct ActualText), got {:?}",
        extracted
    );
}
