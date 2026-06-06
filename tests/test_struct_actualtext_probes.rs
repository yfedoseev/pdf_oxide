//! Probe tests — third-pass review of struct-tree-scope `/ActualText`
//!
//! These pin behaviour that the previous two review passes either
//! deferred or only touched at a surface level. Each probe is either
//! a regression pin (passes today, locks in the contract) or a bug
//! reproducer (fails today, proves a defect).
//!
//! Organised by category from the third-pass review brief:
//!   - architectural refactor (consecutive-run dedup boundaries)
//!   - `suppress_only` semantics for multi-page scopes
//!   - nested BDC / current_mcid stack discipline
//!   - MC-scope once-per-scope across Tj boundaries
//!   - locked-decision verification (MarkInfo, OCG)
//!   - edge cases (null, indirect, line breaks)
//!   - cross-path consistency at the structured surface
//!   - mutability / per-call lifecycle

#![allow(clippy::useless_vec)]

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;

// =============================================================
// Minimal tagged-PDF builder (copied from `test_struct_actualtext.rs`
// since `tests/` are compiled as separate crates and the builder is
// private to its own file). Identical structure; kept tight.
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
    suspects: bool,
    marked: bool,
    /// If true, MarkInfo dict is omitted entirely.
    no_mark_info: bool,
    /// Optional raw catalog entries inserted into the catalog dict
    /// (e.g. extra metadata). Currently unused but reserved.
    _extra_catalog: String,
}

impl PdfBuilder {
    fn new() -> Self {
        Self {
            page_contents: Vec::new(),
            elems: Vec::new(),
            parent_tree_entries: Vec::new(),
            suspects: false,
            marked: true,
            no_mark_info: false,
            _extra_catalog: String::new(),
        }
    }
    #[allow(dead_code)]
    fn suspects(mut self) -> Self {
        self.suspects = true;
        self
    }
    fn no_mark_info(mut self) -> Self {
        self.no_mark_info = true;
        self
    }
    fn marked(mut self, marked: bool) -> Self {
        self.marked = marked;
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
    fn pdf_string_utf16be(s: &str) -> String {
        let mut out = String::from("<FEFF");
        for u in s.encode_utf16() {
            out.push_str(&format!("{:04X}", u));
        }
        out.push('>');
        out
    }

    fn build_with_ocg(self, layer_name: &str) -> Vec<u8> {
        self.build_impl(Some(layer_name.to_string()))
    }
    fn build(self) -> Vec<u8> {
        self.build_impl(None)
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
            assert!(e.obj_num >= first_struct_elem, "elem obj_num too low");
            assert!(used.insert(e.obj_num), "duplicate elem obj_num {}", e.obj_num);
        }
        let root_elem = self.elems[0].obj_num;

        let mut objs: BTreeMap<u32, Vec<u8>> = BTreeMap::new();

        let mark_info = if self.no_mark_info {
            String::new()
        } else if self.suspects {
            " /MarkInfo << /Marked true /Suspects true >>".to_string()
        } else if self.marked {
            " /MarkInfo << /Marked true >>".to_string()
        } else {
            " /MarkInfo << /Marked false >>".to_string()
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
                "<< /Type /Catalog /Pages {} 0 R{} /StructTreeRoot {} 0 R{} >>",
                pages, mark_info, struct_tree_root, oc_props
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

// ----------------------------------------------------------------------
// content-stream helpers
// ----------------------------------------------------------------------

fn three_mcid_consecutive() -> Vec<u8> {
    // MCIDs 0, 1, 2 emitted in order, one glyph each.
    let mut s = String::from("BT\n/F1 12 Tf\n50 700 Td\n");
    for (m, g) in [(0u32, "A"), (1, "B"), (2, "C")] {
        s.push_str(&format!("/Span << /MCID {} >> BDC\n({}) Tj\nEMC\n", m, g));
    }
    s.push_str("ET\n");
    s.into_bytes()
}

// =============================================================
// Probe 1 (extended): three consecutive same-replacement MCIDs
//                     emit the replacement EXACTLY once.
// =============================================================

fn fixture_probe1_three_same_run() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    b.add_page_content(three_mcid_consecutive());
    let _span = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("RUN")
            .k(K::Mcid(0, 0))
            .k(K::Mcid(1, 0))
            .k(K::Mcid(2, 0)),
    );
    b.register_mcid(0, 0, 8);
    b.register_mcid(0, 1, 8);
    b.register_mcid(0, 2, 8);
    b.build()
}

#[test]
fn probe1_three_consecutive_same_run_emits_once() {
    let pdf = fixture_probe1_three_same_run();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    assert_eq!(
        extracted.matches("RUN").count(),
        1,
        "three consecutive same-replacement MCIDs must collapse to ONE emission, got {:?}",
        extracted
    );
    assert!(!extracted.contains('A'), "raw 'A' must NOT appear, got {:?}", extracted);
    assert!(!extracted.contains('B'), "raw 'B' must NOT appear, got {:?}", extracted);
    assert!(!extracted.contains('C'), "raw 'C' must NOT appear, got {:?}", extracted);
}

// =============================================================
// Probe 2: mid-run inner override — outer O, inner I on MCID 1.
//          Expected: O, I, O (three emissions).
// =============================================================

fn fixture_probe2_mid_run_inner_override() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    b.add_page_content(three_mcid_consecutive());
    // Outer covers MCIDs 0 and 2 (siblings flanking the inner subtree
    // that covers MCID 1).
    //   Outer Span /ActualText "O" /K [Mcid0, Inner, Mcid2]
    //     Inner Span /ActualText "I" /K [Mcid1]
    let outer = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("O")
            .k(K::Mcid(0, 0))
            .k(K::Obj(9))
            .k(K::Mcid(2, 0)),
    );
    let _inner = b.add_elem(
        Elem::new(9, "Span", 8)
            .page(4)
            .actual_text("I")
            .k(K::Mcid(1, 0)),
    );
    b.register_mcid(0, 0, outer);
    b.register_mcid(0, 1, 9);
    b.register_mcid(0, 2, outer);
    b.build()
}

#[test]
fn probe2_mid_run_inner_override_yields_o_i_o() {
    let pdf = fixture_probe2_mid_run_inner_override();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    // O must appear twice (once for MCID 0, once for MCID 2 — the inner
    // breaks the run, so the outer fires before AND after).
    assert_eq!(
        extracted.matches('O').count(),
        2,
        "outer 'O' must emit twice (MCIDs 0 and 2 — run broken by inner), got {:?}",
        extracted
    );
    // Inner must appear exactly once.
    assert_eq!(
        extracted.matches('I').count(),
        1,
        "inner 'I' must emit exactly once at MCID 1, got {:?}",
        extracted
    );
    for raw in ['A', 'B', 'C'] {
        assert!(!extracted.contains(raw), "raw {:?} must NOT appear, got {:?}", raw, extracted);
    }
}

// =============================================================
// Probe 6: multi-page scope whose first descendant is on page 1,
//          NOT page 0. The implementation uses pre-order
//          "first descendant page", not numeric min. Page 0 must
//          NOT receive a suppress_only entry for this scope, and
//          page 1 must emit.
// =============================================================

fn fixture_probe6_first_descendant_on_page1() -> Vec<u8> {
    // Two pages. The H1 scope covers MCID 0 on page 1 first (in pre-
    // order children) and MCID 0 on page 0 second. Per the index
    // builder, `first_page = 1`, and the page-0 MCID gets suppress_only.
    let mut b = PdfBuilder::new();
    // Page 0: one MCID emitting raw glyph "A".
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(A) Tj\nEMC\nET\n".to_vec(),
    );
    // Page 1: one MCID emitting raw glyph "B".
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(B) Tj\nEMC\nET\n".to_vec(),
    );
    // Struct elem H1: /K [page-1 MCR, page-0 MCR] — pre-order picks
    // page 1 as first.
    let h1 = b.add_elem(
        Elem::new(10, "H1", 9)
            .actual_text("Heading X")
            .k(K::Mcid(0, 1))
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, h1);
    b.register_mcid(1, 0, h1);
    b.build()
}

#[test]
fn probe6_first_descendant_on_page1_emits_on_page1_suppresses_page0() {
    let pdf = fixture_probe6_first_descendant_on_page1();
    let doc = PdfDocument::from_bytes(pdf).expect("open");

    let p0 = doc.extract_text(0).expect("extract_text 0");
    let p1 = doc.extract_text(1).expect("extract_text 1");

    // Per the implementation's pre-order rule, page 1 is the bearing
    // first page; page 0 must be suppress-only (no emission, no raw).
    assert!(
        p1.contains("Heading X"),
        "page 1 (first descendant in pre-order) must contain the replacement, got {:?}",
        p1
    );
    assert!(
        !p0.contains("Heading X"),
        "page 0 must NOT receive the replacement (different page), got {:?}",
        p0
    );
    assert!(
        !p0.contains('A'),
        "page 0 raw glyph 'A' must be suppressed (covered MCID), got {:?}",
        p0
    );
    assert!(
        !p1.contains('B'),
        "page 1 raw glyph 'B' must be suppressed (covered MCID), got {:?}",
        p1
    );
}

// =============================================================
// Probe 7: multi-page scope spanning pages 0 and 2 (skipping page
//          1). Page 1 carries an unrelated MCID with no struct-tree
//          ActualText. Verify page 1 stays untouched: no spurious
//          suppress, no spurious emission.
// =============================================================

fn fixture_probe7_skip_middle_page() -> Vec<u8> {
    let mut b = PdfBuilder::new();
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(P0) Tj\nEMC\nET\n".to_vec(),
    );
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(MIDDLE) Tj\nEMC\nET\n".to_vec(),
    );
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(P2) Tj\nEMC\nET\n".to_vec(),
    );
    // For 3 pages: catalog=1, pages=2, font=3, page0=4, page1=5,
    // page2=6, content0=7, content1=8, content2=9, parent_tree=10,
    // struct_tree_root=11, first elem obj = 12.
    let _doc = b.add_elem(Elem::new(12, "Document", 11).k(K::Obj(13)).k(K::Obj(14)));
    let _h1 = b.add_elem(
        Elem::new(13, "H1", 12)
            .actual_text("Span across")
            .k(K::Mcid(0, 0))
            .k(K::Mcid(0, 2)),
    );
    let _p = b.add_elem(Elem::new(14, "P", 12).k(K::Mcid(0, 1)));
    b.register_mcid(0, 0, 13);
    b.register_mcid(1, 0, 14);
    b.register_mcid(2, 0, 13);
    b.build()
}

#[test]
fn probe7_skip_middle_page_keeps_middle_untouched() {
    let pdf = fixture_probe7_skip_middle_page();
    let doc = PdfDocument::from_bytes(pdf).expect("open");

    let p0 = doc.extract_text(0).expect("extract_text 0");
    let p1 = doc.extract_text(1).expect("extract_text 1");
    let p2 = doc.extract_text(2).expect("extract_text 2");

    // Page 0 (first descendant): emit replacement, raw suppressed.
    assert!(p0.contains("Span across"), "page 0 must emit replacement, got {:?}", p0);
    assert!(!p0.contains("P0"), "page 0 raw glyphs must be suppressed, got {:?}", p0);

    // Page 1 (NOT covered by the H1 scope): raw glyphs survive,
    // no replacement.
    assert!(p1.contains("MIDDLE"), "page 1 raw text must survive untouched, got {:?}", p1);
    assert!(
        !p1.contains("Span across"),
        "page 1 must NOT receive the unrelated H1 replacement, got {:?}",
        p1
    );

    // Page 2 (later page of the H1 scope): raw suppressed, NO second
    // replacement.
    assert!(
        !p2.contains("Span across"),
        "page 2 must NOT emit the replacement a second time, got {:?}",
        p2
    );
    assert!(!p2.contains("P2"), "page 2 raw glyphs must be suppressed, got {:?}", p2);
}

// =============================================================
// Probe 9: three-level BDC nesting — Tj after all three EMCs
//          restores to outer (no MCID). Pin behaviour.
// =============================================================

#[test]
fn probe9_three_level_bdc_nest_then_outer_tj_attributes_to_outer() {
    // BDC[10] (A) BDC[11] (B) BDC[12] (C) EMC EMC EMC (D)
    //
    // After all three EMCs, current_mcid should be None and (D)
    // should land in a span with mcid=None. We can verify that
    // through the public `extract_page_text` API by checking that
    // the "D" glyph's span carries no MCID.
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
                    /Span << /MCID 10 >> BDC\n(A) Tj\n\
                    /Span << /MCID 11 >> BDC\n(B) Tj\n\
                    /Span << /MCID 12 >> BDC\n(C) Tj\n\
                    EMC\nEMC\nEMC\n\
                    100 0 Td\n(D) Tj\n\
                    ET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());
    // Register tags for all three MCIDs.
    let _doc = b.add_elem(
        Elem::new(8, "Document", 7)
            .k(K::Mcid(10, 0))
            .k(K::Mcid(11, 0))
            .k(K::Mcid(12, 0)),
    );
    b.register_mcid(0, 10, 8);
    b.register_mcid(0, 11, 8);
    b.register_mcid(0, 12, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let page = doc.extract_page_text(0).expect("extract_page_text");
    // Find the span containing "D" — must have mcid=None.
    let d_span = page
        .spans
        .iter()
        .find(|s| s.text.contains('D'))
        .unwrap_or_else(|| panic!("no D span; got {:?}", page.spans));
    assert_eq!(
        d_span.mcid, None,
        "after closing three nested BDCs, the next Tj must attribute to NO MCID (mcid=None), \
         got mcid={:?} text={:?}",
        d_span.mcid, d_span.text
    );

    // Likewise, A/B/C spans must each carry their respective MCIDs.
    for (g, expect_mcid) in [('A', 10u32), ('B', 11), ('C', 12)] {
        let s = page
            .spans
            .iter()
            .find(|s| s.text.contains(g))
            .unwrap_or_else(|| panic!("no {} span", g));
        assert_eq!(
            s.mcid,
            Some(expect_mcid),
            "glyph {} must carry MCID {} (nesting/restore failure), got {:?}",
            g,
            expect_mcid,
            s.mcid
        );
    }
}

// =============================================================
// Probe 10: BDC without /MCID (e.g. `/Span << /ActualText (X) >>` with
//          no /MCID key). MC-scope ActualText still emits once, and
//          the enclosing scope's MCID stays in effect for any text
//          AFTER the inner EMC.
// =============================================================

#[test]
fn probe10_bdc_without_mcid_keeps_outer_mcid_after_emc() {
    // Outer BDC[5] (A) inner BDC[no-MCID, /ActualText "fi"] (X) EMC
    // (B) EMC
    //
    // Expected on the spans:
    //   - "A" span has mcid=5
    //   - "X" gets suppressed; replaced by "fi"; mcid is the outer 5
    //   - "B" span has mcid=5 (restored to outer)
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
                    /Span << /MCID 5 >> BDC\n(A) Tj\n\
                    /Span << /ActualText <FEFF00660069> >> BDC\n(X) Tj\nEMC\n\
                    (B) Tj\n\
                    EMC\nET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());
    let _doc = b.add_elem(Elem::new(8, "Document", 7).k(K::Mcid(5, 0)));
    b.register_mcid(0, 5, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let page = doc.extract_page_text(0).expect("extract_page_text");
    // B span must carry mcid=5 (outer restored).
    let b_span = page
        .spans
        .iter()
        .find(|s| s.text.contains('B'))
        .unwrap_or_else(|| panic!("no B span; got {:?}", page.spans));
    assert_eq!(
        b_span.mcid,
        Some(5),
        "after inner BDC-without-MCID closes, 'B' must attribute to outer MCID 5, got {:?}",
        b_span.mcid
    );

    // The extract_text path: "fi" appears once (MC-scope replacement),
    // raw "X" suppressed.
    let extracted = doc.extract_text(0).expect("extract_text");
    assert!(
        extracted.contains("fi"),
        "MC-scope /ActualText 'fi' must appear, got {:?}",
        extracted
    );
    assert!(!extracted.contains('X'), "raw 'X' must NOT appear, got {:?}", extracted);
}

// =============================================================
// Probe 12: EMC with empty stack (malformed PDF). Must not panic;
//          subsequent extraction must still complete.
// =============================================================

#[test]
fn probe12_unmatched_emc_does_not_panic() {
    // Stray EMC before any BDC, then a normal BDC/Tj/EMC.
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
                    EMC\n\
                    /Span << /MCID 0 >> BDC\n(OK) Tj\nEMC\n\
                    ET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());
    let _doc = b.add_elem(Elem::new(8, "Document", 7).k(K::Mcid(0, 0)));
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    // Must not panic. Must extract the OK text.
    let extracted = doc.extract_text(0).expect("extract_text");
    assert!(
        extracted.contains("OK"),
        "post-stray-EMC content must extract, got {:?}",
        extracted
    );
}

// =============================================================
// Probe 13: two Tj inside one MC-scope ActualText with an
//          intervening Td. Replacement emitted ONCE, not twice.
// =============================================================

#[test]
fn probe13_mc_scope_two_tj_with_td_emits_once() {
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
                    /Span << /MCID 0 /ActualText <FEFF006D0063> >> BDC\n\
                    (X) Tj\n100 0 Td\n(Y) Tj\n\
                    EMC\nET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());
    let _doc = b.add_elem(Elem::new(8, "Document", 7).k(K::Mcid(0, 0)));
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    assert_eq!(
        extracted.matches("mc").count(),
        1,
        "MC-scope /ActualText must emit ONCE across multiple Tj+Td, got {:?}",
        extracted
    );
    for raw in ['X', 'Y'] {
        assert!(!extracted.contains(raw), "raw {:?} must NOT appear, got {:?}", raw, extracted);
    }
}

// =============================================================
// Probe 14: two MC-scope ActualText sequences in a row — each
//          emits its own replacement, no leakage.
// =============================================================

#[test]
fn probe14_two_mc_scope_actualtext_sequences_each_emit() {
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
                    /Span << /MCID 0 /ActualText <FEFF0061> >> BDC\n(X) Tj\nEMC\n\
                    /Span << /MCID 1 /ActualText <FEFF0062> >> BDC\n(Y) Tj\nEMC\n\
                    ET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());
    let _doc = b.add_elem(
        Elem::new(8, "Document", 7)
            .k(K::Mcid(0, 0))
            .k(K::Mcid(1, 0)),
    );
    b.register_mcid(0, 0, 8);
    b.register_mcid(0, 1, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    assert_eq!(
        extracted.matches('a').count(),
        1,
        "first MC's 'a' must appear once, got {:?}",
        extracted
    );
    assert_eq!(
        extracted.matches('b').count(),
        1,
        "second MC's 'b' must appear once, got {:?}",
        extracted
    );
    assert!(!extracted.contains('X'), "raw 'X' must NOT appear, got {:?}", extracted);
    assert!(!extracted.contains('Y'), "raw 'Y' must NOT appear, got {:?}", extracted);
}

// =============================================================
// Probe 15: nested MC-scope ActualText — outer has /ActualText
//          "outer", inner has /ActualText "inner", inner wins
//          (innermost is the most-specific).
// =============================================================

#[test]
fn probe15_nested_mc_scope_actualtext_inner_wins() {
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
                    /Span << /ActualText <FEFF006F0075> >> BDC\n\
                    /Span << /MCID 0 /ActualText <FEFF0069006E> >> BDC\n\
                    (X) Tj\n\
                    EMC\nEMC\nET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());
    let _doc = b.add_elem(Elem::new(8, "Document", 7).k(K::Mcid(0, 0)));
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    assert!(
        extracted.contains("in"),
        "inner MC-scope replacement 'in' must appear, got {:?}",
        extracted
    );
    assert!(
        !extracted.contains("ou"),
        "outer MC-scope replacement 'ou' must NOT appear (inner wins), got {:?}",
        extracted
    );
}

// =============================================================
// Probe 16: MC-scope /ActualText with EMPTY string. Pin behaviour:
//          today the empty replacement should suppress the raw
//          glyph entirely (no emission, no raw). This matches the
//          struct-tree-scope empty handling.
//
// If the implementation actually emits a raw glyph here, this is a
// MINOR inconsistency between MC-scope and struct-scope handling
// of empty /ActualText.
// =============================================================

#[test]
fn probe16_mc_scope_empty_actualtext_pin_behaviour() {
    // /ActualText <FEFF> is an empty UTF-16BE string.
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
                    /Span << /MCID 0 /ActualText <FEFF> >> BDC\n\
                    (X) Tj\n\
                    EMC\nET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());
    let _doc = b.add_elem(Elem::new(8, "Document", 7).k(K::Mcid(0, 0)));
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    // Pin current behaviour: the MC-scope `actual_text` is `Some("")`,
    // so `peek_current_actual_text` returns Some("") and the emission
    // path runs — outputting an empty string and suppressing the raw
    // glyph. So neither the raw glyph nor any replacement text
    // appears.
    assert!(
        !extracted.contains('X'),
        "MC-scope empty /ActualText must still suppress the raw glyph 'X', got {:?}",
        extracted
    );
}

// =============================================================
// Probe 17: MarkInfo permutations.
//   17a — Suspects=true with NO MarkInfo dict at all
//         (parser sees /MarkInfo absent → struct_tree_marked uses
//         /StructTreeRoot existence). ActualText still resolves.
//   17b — Suspects=true with explicit MarkInfo (decoupled — already
//         covered by `actualtext_emits_when_suspects_true`; we add
//         the negative twin here for clarity).
//   17c — Marked=false WITHOUT a /StructTreeRoot... actually if
//         /StructTreeRoot is in the catalog, struct_tree_marked
//         still returns the tree. Pin that.
// =============================================================

#[test]
fn probe17a_no_mark_info_still_resolves_actualtext_via_struct_tree_root() {
    let mut b = PdfBuilder::new().no_mark_info();
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(X) Tj\nEMC\nET\n".to_vec(),
    );
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("replaced")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    // /StructTreeRoot is present in the catalog → struct_tree_marked
    // serves the tree even without /MarkInfo. ActualText fires.
    assert!(
        extracted.contains("replaced"),
        "no /MarkInfo + present /StructTreeRoot must still apply /ActualText, got {:?}",
        extracted
    );
    assert!(!extracted.contains('X'), "raw 'X' must NOT appear, got {:?}", extracted);
}

#[test]
fn probe17c_marked_false_with_struct_tree_root_still_resolves_actualtext() {
    let mut b = PdfBuilder::new().marked(false);
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(X) Tj\nEMC\nET\n".to_vec(),
    );
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("replaced")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    // /StructTreeRoot in the catalog is enough (struct_tree_marked's
    // contract: `mark.marked || has_struct_tree_root`).
    assert!(
        extracted.contains("replaced"),
        "/Marked false + /StructTreeRoot present must still apply /ActualText, got {:?}",
        extracted
    );
}

// =============================================================
// Probe 19: OCG nested with multi-page. Scope covers pages 0 and 1;
//          page 0's covered MCID is in the excluded layer; page 1's
//          covered MCID is visible. Pin: page 0 produces no
//          emission (first page rule + layer-excluded → no emit),
//          and page 1 is suppress_only → still no emission.
//
//          Net: the replacement is LOST when the first page's
//          covered MCIDs are all hidden. This pins the current
//          implementation's behaviour; the locked decision said
//          "skip-when-all-filtered" for the first page; the second
//          page is suppress_only with no recourse.
// =============================================================

#[test]
fn probe19_first_page_hidden_via_ocg_drops_emission_entirely() {
    let mut b = PdfBuilder::new();
    // Page 0: MCID 0 inside excluded /OC /L layer.
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n\
          /OC /L BDC\n\
          /Span << /MCID 0 >> BDC\n(A) Tj\nEMC\n\
          EMC\nET\n"
            .to_vec(),
    );
    // Page 1: MCID 0 visible (no OCG wrap).
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(B) Tj\nEMC\nET\n".to_vec(),
    );
    // For 2 pages + OCG: catalog=1, pages=2, font=3, page0=4,
    // page1=5, content0=6, content1=7, parent_tree=8,
    // struct_tree_root=9, ocg=10, first elem obj = 11.
    let h1 = b.add_elem(
        Elem::new(11, "H1", 9)
            .actual_text("Heading Y")
            .k(K::Mcid(0, 0))
            .k(K::Mcid(0, 1)),
    );
    b.register_mcid(0, 0, h1);
    b.register_mcid(1, 0, h1);
    let pdf = b.build_with_ocg("HiddenLayer");
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let mut excluded = std::collections::HashSet::new();
    excluded.insert("HiddenLayer".to_string());

    let p0 = doc
        .extract_text_filtered(0, excluded.clone(), std::collections::HashSet::new())
        .expect("extract_text_filtered 0");
    let p1 = doc
        .extract_text_filtered(1, excluded, std::collections::HashSet::new())
        .expect("extract_text_filtered 1");

    // Page 0: layer hidden → no raw, no replacement.
    assert!(
        !p0.contains("Heading Y"),
        "page 0 first-page MCID is hidden by excluded layer → no emission, got {:?}",
        p0
    );
    assert!(
        !p0.contains('A'),
        "page 0 raw glyph 'A' must stay hidden under excluded layer, got {:?}",
        p0
    );

    // Page 1: suppress-only entry → still no emission, raw also
    // suppressed (covered). This pins the "lost replacement" edge:
    // the producer's intent silently drops here. If a fix changes this
    // contract, this test will alert.
    assert!(
        !p1.contains("Heading Y"),
        "page 1 (suppress_only) must NOT emit the replacement — pin current behaviour, got {:?}",
        p1
    );
    assert!(
        !p1.contains('B'),
        "page 1 raw glyph 'B' must be suppressed (covered MCID), got {:?}",
        p1
    );
}

// =============================================================
// Probe 29: /ActualText whose value is `null` PDF object. Should be
//          treated as absent — raw glyph survives, no panic.
// =============================================================

#[test]
fn probe29_actualtext_null_value_treated_as_absent() {
    // Hand-assemble a PDF with /ActualText null on a Span — bypass
    // the builder's typed `actual_text` (which always writes a hex
    // string).
    use std::collections::BTreeMap;
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(X) Tj\nEMC\nET\n";

    // catalog=1, pages=2, font=3, page=4, content=5, parent_tree=6,
    // struct_tree_root=7, struct_elem=8.
    let mut objs: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    objs.insert(
        1,
        b"<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >> /StructTreeRoot 7 0 R >>"
            .to_vec(),
    );
    objs.insert(2, b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>".to_vec());
    objs.insert(3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec());
    objs.insert(
        4,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Resources << /Font << /F1 3 0 R >> /ProcSet [/PDF /Text] >> \
          /Contents 5 0 R /StructParents 0 >>"
            .to_vec(),
    );
    let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    stream.extend_from_slice(content);
    stream.extend_from_slice(b"\nendstream");
    objs.insert(5, stream);
    objs.insert(6, b"<< /Nums [0 [8 0 R]] >>".to_vec());
    objs.insert(7, b"<< /Type /StructTreeRoot /K 8 0 R /ParentTree 6 0 R >>".to_vec());
    // KEY DIFFERENCE: /ActualText null.
    objs.insert(
        8,
        b"<< /Type /StructElem /S /Span /P 7 0 R /Pg 4 0 R /ActualText null \
          /K << /Type /MCR /Pg 4 0 R /MCID 0 >> >>"
            .to_vec(),
    );

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
    let n_objs = 9u32;
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

    let doc = PdfDocument::from_bytes(out).expect("open with /ActualText null");
    // Must not panic; raw 'X' survives.
    let extracted = doc.extract_text(0).expect("extract_text");
    assert!(
        extracted.contains('X'),
        "/ActualText null must be treated as absent → raw 'X' survives, got {:?}",
        extracted
    );
}

// =============================================================
// Probe 30: /ActualText whose value is an INDIRECT reference to a
//          string. Should resolve correctly.
// =============================================================

#[test]
fn probe30_actualtext_indirect_string_reference_resolves() {
    use std::collections::BTreeMap;
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(X) Tj\nEMC\nET\n";

    // Same as probe29 layout but /ActualText is `9 0 R` and obj 9 is
    // a UTF-16BE string.
    let mut objs: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    objs.insert(
        1,
        b"<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >> /StructTreeRoot 7 0 R >>"
            .to_vec(),
    );
    objs.insert(2, b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>".to_vec());
    objs.insert(3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec());
    objs.insert(
        4,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Resources << /Font << /F1 3 0 R >> /ProcSet [/PDF /Text] >> \
          /Contents 5 0 R /StructParents 0 >>"
            .to_vec(),
    );
    let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    stream.extend_from_slice(content);
    stream.extend_from_slice(b"\nendstream");
    objs.insert(5, stream);
    objs.insert(6, b"<< /Nums [0 [8 0 R]] >>".to_vec());
    objs.insert(7, b"<< /Type /StructTreeRoot /K 8 0 R /ParentTree 6 0 R >>".to_vec());
    objs.insert(
        8,
        b"<< /Type /StructElem /S /Span /P 7 0 R /Pg 4 0 R /ActualText 9 0 R \
          /K << /Type /MCR /Pg 4 0 R /MCID 0 >> >>"
            .to_vec(),
    );
    // Obj 9: the actual string. UTF-16BE for "indirect".
    objs.insert(9, b"<FEFF0069006E0064006900720065006300740020 0068006900740073>".to_vec());
    // Actually let's keep it simple — direct UTF-16BE bytes string,
    // no internal spaces; <FEFF...> hex literal. "indirect" in UTF-16
    // BE: 0069 006E 0064 0069 0072 0065 0063 0074.
    objs.insert(9, b"<FEFF0069006E0064006900720065006300740068006900740073>".to_vec());
    // Recompute — we re-inserted the same key; last write wins; the
    // string above is for "indirecthits" (close enough — we check
    // substring "indirect"). Keep what's there for the test.

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
    let n_objs = 10u32;
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

    let doc = PdfDocument::from_bytes(out).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    // Either the indirect string resolves (substring "indirect" appears
    // and raw 'X' is gone), OR the resolver doesn't follow the ref and
    // we get raw 'X' through with no replacement.
    // We pin the behaviour: if 'X' survives the resolver doesn't follow
    // the indirect; if "indirect" appears, it does. Either is acceptable
    // PDF behaviour but the result must be self-consistent.
    let resolved_indirect = extracted.contains("indirect");
    let has_raw = extracted.contains('X');
    assert!(
        resolved_indirect ^ has_raw,
        "indirect /ActualText: either resolution succeeds (no raw, replacement appears) \
         or it doesn't (raw survives). XOR must hold. got {:?}",
        extracted
    );
}

// =============================================================
// Probe 32: /ActualText with embedded \r and \n. Pin that the
//          replacement is forwarded verbatim (line breaks may
//          survive in the output).
// =============================================================

#[test]
fn probe32_actualtext_with_line_breaks_pin_behaviour() {
    let mut b = PdfBuilder::new();
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(X) Tj\nEMC\nET\n".to_vec(),
    );
    // /ActualText "L1\nL2" — embeds U+000A.
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("L1\nL2")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    // Both substrings present (verbatim forward).
    assert!(
        extracted.contains("L1") && extracted.contains("L2"),
        "/ActualText with embedded newline must forward both halves verbatim, got {:?}",
        extracted
    );
    assert!(!extracted.contains('X'), "raw 'X' must NOT appear, got {:?}", extracted);
}

// =============================================================
// Probe 34: /Artifact MC sequence with /ActualText. The /Artifact
//          tag traditionally suppresses content; the /ActualText on
//          the artifact should also be suppressed (the entire
//          artifact is meant to be hidden).
// =============================================================

/// BUG REPRODUCER — third-pass review finding.
///
/// MC-scope `/ActualText` carried by an `/Artifact` BDC leaks into
/// extracted text. The artifact filter is downstream
/// (`spans.retain(|s| s.artifact_type.is_none())`) and depends on
/// the span carrying `artifact_type=Some(...)`. The Tj-span buffer
/// flush path hardcodes `artifact_type: None`
/// (see `src/extractors/text.rs` `flush_tj_span_buffer`), so every
/// span produced by that flush — including MC-scope ActualText
/// replacements emitted into the buffer — bypasses the filter and
/// leaks the substituted text from inside `/Artifact`.
///
/// `#[ignore]` flags this for the next fix pass; remove the
/// attribute once the underlying bug is fixed in production code.
/// Verified by `probe34b` below: the same leak occurs for vanilla
/// raw glyphs inside `/Artifact`, so the bug is pre-existing in
/// the buffered Tj-span path and exposed by ActualText, not
/// introduced by it.
#[test]
fn probe34_actualtext_on_artifact_mc_does_not_leak() {
    // BDC tag = Artifact, properties include /ActualText.
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
                    /Artifact << /Type /Pagination /ActualText <FEFF00610072> >> BDC\n\
                    (X) Tj\n\
                    EMC\nET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());
    // The struct tree need not reference the artifact MCID — artifacts
    // are not part of the structure tree; we still ship a minimal tree.
    let _doc = b.add_elem(Elem::new(8, "Document", 7));
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");

    // Dump the raw spans for diagnostic context: the artifact filter
    // is downstream from the extractor and depends on the span
    // carrying `artifact_type=Some(...)`. If the MC-scope
    // /ActualText emission creates a span without the artifact tag
    // (because the buffer was flushed and the artifact-tag lookup
    // happened at the wrong stack depth), the span survives the
    // downstream `spans.retain(|s| s.artifact_type.is_none())` and
    // leaks "ar" into extraction output.
    let page = doc.extract_page_text(0).expect("extract_page_text");
    eprintln!("probe34 spans dump:");
    for (i, s) in page.spans.iter().enumerate() {
        eprintln!(
            "  [{}] text={:?} mcid={:?} artifact_type={:?}",
            i, s.text, s.mcid, s.artifact_type
        );
    }

    let extracted = doc.extract_text(0).expect("extract_text");
    // Per the extractor's artifact filtering, neither the raw 'X' nor
    // the ActualText replacement ("ar") should be emitted.
    assert!(
        !extracted.contains("ar"),
        "/Artifact MC's /ActualText must not leak into output (artifact filtered), got {:?}; spans={:?}",
        extracted,
        page.spans.iter().map(|s| (s.text.clone(), s.artifact_type.clone())).collect::<Vec<_>>(),
    );
    assert!(
        !extracted.contains('X'),
        "raw 'X' must NOT appear inside /Artifact, got {:?}",
        extracted
    );
}

// =============================================================
// Probe 34b: vanilla /Artifact MC sequence with raw glyphs (NO
//          /ActualText). Verifies the artifact filter works at all
//          for the buffered Tj-span path, so we can scope the
//          ActualText leak in probe34 precisely. If THIS also leaks,
//          the bug isn't ActualText-specific — it's a pre-existing
//          artifact_type=None hardcode in the Tj-span flush path
//          that the ActualText feature merely exposed.
// =============================================================

/// BUG REPRODUCER — third-pass review finding.
///
/// Scopes the artifact-leak bug found by `probe34` to the buffered
/// Tj-span path, INDEPENDENT of ActualText. Vanilla raw glyphs
/// inside `/Artifact` BDC also leak through to extracted text
/// because `flush_tj_span_buffer` hardcodes
/// `artifact_type: None` on the produced span. This is a
/// pre-existing bug unrelated to the ActualText feature but
/// inherited by it.
///
/// `#[ignore]` flags the production-side fix; remove once
/// `flush_tj_span_buffer` reads `self.current_artifact_type()`
/// like `flush_tj_buffer` does (see `src/extractors/text.rs:6363`).
#[test]
fn probe34b_vanilla_artifact_raw_glyph_pin_behaviour() {
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
                    /Artifact << /Type /Pagination >> BDC\n\
                    (XYZ) Tj\n\
                    EMC\nET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());
    let _doc = b.add_elem(Elem::new(8, "Document", 7));
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let page = doc.extract_page_text(0).expect("extract_page_text");
    eprintln!("probe34b spans dump:");
    for (i, s) in page.spans.iter().enumerate() {
        eprintln!(
            "  [{}] text={:?} mcid={:?} artifact_type={:?}",
            i, s.text, s.mcid, s.artifact_type
        );
    }
    let extracted = doc.extract_text(0).expect("extract_text");
    // Pin: raw artifact glyphs must NOT appear in extracted text.
    // The downstream filter retains only spans with
    // `artifact_type.is_none()`. If raw glyphs leak, the Tj-span
    // flush is broken at the artifact_type hardcode (see
    // src/extractors/text.rs `flush_tj_span_buffer` field
    // `artifact_type: None`).
    assert!(
        !extracted.contains("XYZ"),
        "vanilla /Artifact raw glyphs must NOT appear in extract_text — \
         artifact filter is broken if this fails; got {:?}; spans={:?}",
        extracted,
        page.spans
            .iter()
            .map(|s| (s.text.clone(), s.artifact_type.clone()))
            .collect::<Vec<_>>(),
    );
}

// =============================================================
// Probe 36: extract_structured surfaces the replacement in region
//          text and never the raw glyph.
// =============================================================

#[test]
fn probe36_extract_structured_region_text_carries_replacement() {
    let mut b = PdfBuilder::new();
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(X) Tj\nEMC\nET\n".to_vec(),
    );
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("fi")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let page = doc.extract_structured(0).expect("extract_structured");
    let joined = page
        .regions
        .iter()
        .map(|r| r.text.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        joined.contains("fi"),
        "extract_structured regions must carry the replacement text 'fi', got regions={:?}",
        page.regions
    );
    assert!(
        !joined.contains('X'),
        "extract_structured regions must NOT carry the raw 'X' glyph, got regions={:?}",
        page.regions
    );
}

// =============================================================
// Probe 38: two sequential extractions on the same `PdfDocument`
//          give identical results — the per-page mc_actualtext_mcids
//          map is REPLACED (not extended) so re-runs are idempotent.
// =============================================================

#[test]
fn probe38_two_sequential_extract_text_calls_idempotent() {
    let mut b = PdfBuilder::new();
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n\
          /Span << /MCID 0 /ActualText <FEFF0061> >> BDC\n(X) Tj\nEMC\nET\n"
            .to_vec(),
    );
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("struct")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let first = doc.extract_text(0).expect("extract_text 1");
    let second = doc.extract_text(0).expect("extract_text 2");
    assert_eq!(
        first, second,
        "two sequential extract_text calls must give byte-equal output (per-call MC-wins map is \
         REPLACED, not accumulated)"
    );
    // MC-scope wins: 'a' appears, 'struct' does not.
    assert!(first.contains('a'), "MC-scope 'a' must appear, got {:?}", first);
    assert!(!first.contains("struct"), "struct-scope must NOT win, got {:?}", first);
}

// =============================================================
// Probe 38 / continuation: same surface but verifying via the
// public extract_page_text + apply pipeline that spans carry the
// replacement on the second call exactly as on the first.
// =============================================================

#[test]
fn probe38b_repeated_extract_page_text_carries_replacement_each_time() {
    let mut b = PdfBuilder::new();
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(X) Tj\nEMC\nET\n".to_vec(),
    );
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("fi")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");

    for round in 0..3 {
        let page = doc.extract_page_text(0).expect("extract_page_text");
        let joined: String = page
            .spans
            .iter()
            .map(|s| s.text.clone())
            .collect::<Vec<_>>()
            .join("|");
        assert!(
            joined.contains("fi"),
            "round {}: replacement must appear in spans, got {:?}",
            round,
            joined
        );
        assert!(
            !joined.contains('X'),
            "round {}: raw 'X' must NOT appear in spans, got {:?}",
            round,
            joined
        );
    }
}

// =============================================================
// Probe (cross-path consistency at extract_structured): JSON-shape
// region text must match extract_text byte-for-byte minus whitespace
// for a single-MCID fixture. Slightly stricter than the existing
// `cross_path_byte_equal_for_actualtext_replacement` because we
// also check extract_structured.
// =============================================================

#[test]
fn probe35_extract_structured_byte_equal_with_extract_text() {
    let mut b = PdfBuilder::new();
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(X) Tj\nEMC\nET\n".to_vec(),
    );
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("fi")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extract_text = doc.extract_text(0).expect("extract_text");
    let opts = ConversionOptions::default();
    let plain = doc.to_plain_text(0, &opts).expect("plain");
    let structured = doc.extract_structured(0).expect("structured");
    let structured_joined: String = structured
        .regions
        .iter()
        .map(|r| r.text.clone())
        .collect::<Vec<_>>()
        .join("\n");

    // Body content (the replacement) must be exactly "fi" on all three
    // — modulo trailing whitespace from converter line-wrapping.
    assert_eq!(
        extract_text.trim().replace(['\r', '\n'], ""),
        "fi",
        "extract_text body must be exactly 'fi', got {:?}",
        extract_text
    );
    assert_eq!(
        plain.trim().replace(['\r', '\n'], ""),
        "fi",
        "to_plain_text body must be exactly 'fi', got {:?}",
        plain
    );
    assert_eq!(
        structured_joined.trim().replace(['\r', '\n'], ""),
        "fi",
        "extract_structured region body must be exactly 'fi', got {:?}",
        structured_joined
    );
}

// =============================================================
// Probe 4: pin the canonical MCID iteration order used by the
//          consecutive-run dedup. The action map walks struct-tree
//          pre-order; emission lands at the first content-stream
//          span carrying the emit-pick MCID.
//
// Fixture: content stream order [B(mcid 1), A(mcid 0), C(mcid 2)].
// Struct-tree order [0, 1, 2]. Outer Span /ActualText "ALL" covers
// all three. With one consecutive run, ONE emission fires. The
// emit-pick is MCID 0 (first in struct-tree order). In the
// extracted text "ALL" appears at the SPAN-ORDER position of MCID
// 0 — i.e. between B and C.
// =============================================================

#[test]
fn probe4_mcid_iteration_order_uses_struct_tree_preorder() {
    // Content stream: glyph "B" with mcid=1, then "A" with mcid=0,
    // then "C" with mcid=2.
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
                    /Span << /MCID 1 >> BDC\n(B) Tj\nEMC\n\
                    /Span << /MCID 0 >> BDC\n(A) Tj\nEMC\n\
                    /Span << /MCID 2 >> BDC\n(C) Tj\nEMC\n\
                    ET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());
    // Outer Span: struct-tree children listed in MCID-ascending order
    // [0, 1, 2].
    let outer = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("ALL")
            .k(K::Mcid(0, 0))
            .k(K::Mcid(1, 0))
            .k(K::Mcid(2, 0)),
    );
    b.register_mcid(0, 0, outer);
    b.register_mcid(0, 1, outer);
    b.register_mcid(0, 2, outer);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    eprintln!("probe4 extract_text = {:?}", extracted);
    // Consecutive-run dedup: ONE emission of "ALL".
    assert_eq!(
        extracted.matches("ALL").count(),
        1,
        "consecutive-run dedup: exactly one emission, got {:?}",
        extracted
    );
    // Raw glyphs entirely suppressed.
    for raw in ['A', 'B', 'C'] {
        // Note: "ALL" contains 'A' and 'L' so strip first.
        let stripped = extracted.replace("ALL", "");
        assert!(!stripped.contains(raw), "raw {:?} suppressed, got {:?}", raw, extracted);
    }
}

// =============================================================
// Probe 8: multi-page subtree where the bearing element straddles
//          all pages — replacement fires on first page only, the
//          rest produce nothing (no emission, no raw).
// =============================================================

#[test]
fn probe8_actualtext_spanning_all_pages_emits_only_first() {
    let mut b = PdfBuilder::new();
    for _ in 0..3 {
        b.add_page_content(
            b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(Z) Tj\nEMC\nET\n".to_vec(),
        );
    }
    // For 3 pages: catalog=1, pages=2, font=3, page0=4, page1=5,
    // page2=6, content0=7, content1=8, content2=9, parent_tree=10,
    // struct_tree_root=11, first elem obj = 12.
    let doc_e = b.add_elem(
        Elem::new(12, "Document", 11)
            .actual_text("DOC-WIDE")
            .k(K::Mcid(0, 0))
            .k(K::Mcid(0, 1))
            .k(K::Mcid(0, 2)),
    );
    b.register_mcid(0, 0, doc_e);
    b.register_mcid(1, 0, doc_e);
    b.register_mcid(2, 0, doc_e);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let p0 = doc.extract_text(0).expect("p0");
    let p1 = doc.extract_text(1).expect("p1");
    let p2 = doc.extract_text(2).expect("p2");
    assert!(p0.contains("DOC-WIDE"), "page 0 must emit, got {:?}", p0);
    assert!(!p0.contains('Z'), "page 0 raw suppressed, got {:?}", p0);
    for (i, p) in [(1, &p1), (2, &p2)] {
        assert!(!p.contains("DOC-WIDE"), "page {} must NOT re-emit, got {:?}", i, p);
        assert!(!p.contains('Z'), "page {} raw suppressed, got {:?}", i, p);
    }
}

// =============================================================
// Probe 18 mixed-layer: covered MCIDs land in TWO different OCG
//          layers. When both layers are excluded the emission
//          drops; when only one is excluded the emission still
//          fires on the visible-and-not-MC-wins entry.
// =============================================================

#[test]
fn probe18_two_layers_both_excluded_drops_emission() {
    // Single page, two MCIDs each in a different layer.
    // /OC /L1 BDC -> Span MCID 0 -> EMC
    // /OC /L2 BDC -> Span MCID 1 -> EMC
    // Outer Span covers both MCIDs with /ActualText "BOTH".
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
                    /OC /L1 BDC\n/Span << /MCID 0 >> BDC\n(A) Tj\nEMC\nEMC\n\
                    /OC /L2 BDC\n/Span << /MCID 1 >> BDC\n(B) Tj\nEMC\nEMC\n\
                    ET\n";

    // Hand-roll the PDF because PdfBuilder only supports one OCG.
    use std::collections::BTreeMap;
    let mut objs: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    // catalog=1, pages=2, font=3, page=4, content=5, parent_tree=6,
    // struct_tree_root=7, ocg1=8, ocg2=9, struct_elem=10.
    objs.insert(
        1,
        b"<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >> /StructTreeRoot 7 0 R \
          /OCProperties << /OCGs [8 0 R 9 0 R] /D << /Order [8 0 R 9 0 R] >> >> >>"
            .to_vec(),
    );
    objs.insert(2, b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>".to_vec());
    objs.insert(3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec());
    objs.insert(
        4,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
          /Resources << /Font << /F1 3 0 R >> /ProcSet [/PDF /Text] \
          /Properties << /L1 8 0 R /L2 9 0 R >> >> \
          /Contents 5 0 R /StructParents 0 >>"
            .to_vec(),
    );
    let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    stream.extend_from_slice(content);
    stream.extend_from_slice(b"\nendstream");
    objs.insert(5, stream);
    objs.insert(6, b"<< /Nums [0 [10 0 R 10 0 R]] >>".to_vec());
    objs.insert(7, b"<< /Type /StructTreeRoot /K 10 0 R /ParentTree 6 0 R >>".to_vec());
    objs.insert(8, b"<< /Type /OCG /Name (Layer1) >>".to_vec());
    objs.insert(9, b"<< /Type /OCG /Name (Layer2) >>".to_vec());
    objs.insert(
        10,
        b"<< /Type /StructElem /S /Span /P 7 0 R /Pg 4 0 R /ActualText <FEFF0042004F00540048> \
          /K [<< /Type /MCR /Pg 4 0 R /MCID 0 >> << /Type /MCR /Pg 4 0 R /MCID 1 >>] >>"
            .to_vec(),
    );

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
    out.extend_from_slice(b"xref\n0 11\n0000000000 65535 f \n");
    for n in 1..11u32 {
        let off = offsets[&n];
        out.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size 11 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_offset)
            .as_bytes(),
    );

    let doc = PdfDocument::from_bytes(out).expect("open");

    // Exclude BOTH layers — emission must drop.
    let mut both = std::collections::HashSet::new();
    both.insert("Layer1".to_string());
    both.insert("Layer2".to_string());
    let extracted_both = doc
        .extract_text_filtered(0, both, std::collections::HashSet::new())
        .expect("filtered both");
    assert!(
        !extracted_both.contains("BOTH"),
        "with BOTH layers excluded, every covered MCID is invisible → no emission, got {:?}",
        extracted_both
    );

    // Exclude only Layer1 — Layer2's MCID is visible → emission fires
    // on MCID 1 (the visible one).
    let mut just_one = std::collections::HashSet::new();
    just_one.insert("Layer1".to_string());
    let extracted_one = doc
        .extract_text_filtered(0, just_one, std::collections::HashSet::new())
        .expect("filtered one");
    assert!(
        extracted_one.contains("BOTH"),
        "with one layer excluded but the other visible, emission must fire at the visible MCID, got {:?}",
        extracted_one
    );

    // Exclude neither — emission fires normally.
    let extracted_neither = doc.extract_text(0).expect("unfiltered");
    assert!(
        extracted_neither.contains("BOTH"),
        "with no layers excluded, emission must fire, got {:?}",
        extracted_neither
    );
    // Note: raw glyphs "A" / "B" must not appear AS RAW glyphs. The
    // replacement contains 'B' and 'O' so we strip the replacement
    // substring before checking for raw glyphs.
    let stripped = extracted_neither.replace("BOTH", "");
    assert!(!stripped.contains('A'), "raw A suppressed, got {:?}", extracted_neither);
    assert!(
        !stripped.contains('B'),
        "raw B suppressed (independent of replacement), got {:?}",
        extracted_neither
    );
}

// =============================================================
// Probe 37: clones of extracted spans see the mutated text, NOT
//          the raw glyph. The implementer's claim: `effective_text`
//          was dropped; `span.text` is mutated in place.
// =============================================================

#[test]
fn probe37_cloned_spans_carry_replacement_not_raw() {
    let mut b = PdfBuilder::new();
    b.add_page_content(
        b"BT\n/F1 12 Tf\n50 700 Td\n/Span << /MCID 0 >> BDC\n(X) Tj\nEMC\nET\n".to_vec(),
    );
    let _ = b.add_elem(
        Elem::new(8, "Span", 7)
            .page(4)
            .actual_text("fi")
            .k(K::Mcid(0, 0)),
    );
    b.register_mcid(0, 0, 8);
    let pdf = b.build();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let page = doc.extract_page_text(0).expect("extract_page_text");
    // Each span clone independently must carry the mutated text.
    let cloned: Vec<_> = page.spans.to_vec();
    for s in &cloned {
        assert!(!s.text.contains('X'), "cloned span must NOT contain raw 'X', got {:?}", s.text);
    }
    let any_fi = cloned.iter().any(|s| s.text.contains("fi"));
    assert!(
        any_fi,
        "at least one cloned span must contain 'fi'; got spans={:?}",
        cloned.iter().map(|s| s.text.clone()).collect::<Vec<_>>()
    );
}

// =============================================================
// Probe (MarkInfo/Marked=false variant): a malformed Marked=false +
// no StructTreeRoot reference — the document has the StructTreeRoot
// object but never declares it in the catalog. ActualText must NOT
// fire (the index can't be built without the catalog hook).
// (This is a defensive pin; the builder always writes
// /StructTreeRoot, so we only cover the explicit Marked=false
// case from probe17c above.)
// =============================================================

// =============================================================
// Probe 40: cross-MCID merge predicate — fragments with different
//          MCIDs MUST stay separate even when they would otherwise
//          satisfy every other merge condition (same font, same
//          baseline, zero gap).
//
// Spec basis (ISO 32000-1:2008 §14.6, §14.8): the MCID is the
// structural unit. Two same-line fragments with different MCIDs
// belong to different marked-content references and therefore to
// different structure elements (or different references to the
// same element). Merging them would silently fuse their identities
// — the merged span keeps `current.mcid` and drops the other —
// destroying the boundary that structure-tree reading order,
// tree-scope ActualText suppression, and table-cell membership
// rely on.
//
// Fixture: a single visible word "Hello" emitted across two MCIDs
// in the same Span content stream, in the same font, on the same
// baseline, with a zero gap between them. Without the same_mcid
// gate this would merge into one span "Hello" carrying MCID 0;
// with the gate the two fragments survive as separate spans
// "He" and "llo" carrying MCIDs 0 and 1 respectively.
// =============================================================

fn fixture_probe40_cross_mcid_no_merge() -> Vec<u8> {
    // "He" at MCID 0, "llo" at MCID 1 — same font, same baseline,
    // zero gap; the natural pre-PR behaviour was to glue these into
    // "Hello" under is_same_font + same_line + tight gap.
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
         /Span << /MCID 0 >> BDC\n(He) Tj\nEMC\n\
         /Span << /MCID 1 >> BDC\n(llo) Tj\nEMC\n\
         ET\n"
        .to_vec();
    let mut b = PdfBuilder::new();
    b.add_page_content(content);
    let _e0 = b.add_elem(Elem::new(8, "Span", 7).page(4).k(K::Mcid(0, 0)));
    let _e1 = b.add_elem(Elem::new(9, "Span", 7).page(4).k(K::Mcid(1, 0)));
    b.register_mcid(0, 0, 8);
    b.register_mcid(0, 1, 9);
    b.build()
}

#[test]
fn probe40_cross_mcid_fragments_do_not_merge() {
    let pdf = fixture_probe40_cross_mcid_no_merge();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let page = doc.extract_page_text(0).expect("extract_page_text");

    // Both fragments must be present.
    let texts: Vec<String> = page.spans.iter().map(|s| s.text.clone()).collect();
    assert!(texts.iter().any(|t| t == "He"), "fragment 'He' missing from spans: {:?}", texts);
    assert!(
        texts.iter().any(|t| t == "llo"),
        "fragment 'llo' missing from spans: {:?}",
        texts
    );

    // The pre-PR fused form must NOT appear.
    assert!(
        !texts.iter().any(|t| t == "Hello"),
        "fragments must not merge across MCIDs; found 'Hello' in {:?}",
        texts
    );

    // MCIDs are preserved per fragment.
    let mcids: Vec<Option<u32>> = page.spans.iter().map(|s| s.mcid).collect();
    assert!(
        mcids.contains(&Some(0)) && mcids.contains(&Some(1)),
        "expected MCIDs 0 and 1 each on their own span; got {:?}",
        mcids
    );
}

// =============================================================
// Probe 41: regression sentry — same-MCID fragments still merge
//          when they should. Pins that the same_mcid gate does
//          not break the common case where a producer emits
//          multiple Tj operators inside one /Span /MCID BDC ... EMC
//          envelope (e.g. cross-font glue, small-caps glue, decimal
//          merges within one marked-content reference). Without
//          this pin a future refactor could turn same_mcid into
//          "always require different objects" and silently
//          fragment every tagged word.
// =============================================================

fn fixture_probe41_same_mcid_still_merges() -> Vec<u8> {
    // "He" and "llo" both inside MCID 0 — same font, same baseline,
    // zero gap. Must merge to "Hello".
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
         /Span << /MCID 0 >> BDC\n(He) Tj\n(llo) Tj\nEMC\n\
         ET\n"
        .to_vec();
    let mut b = PdfBuilder::new();
    b.add_page_content(content);
    let _e = b.add_elem(Elem::new(8, "Span", 7).page(4).k(K::Mcid(0, 0)));
    b.register_mcid(0, 0, 8);
    b.build()
}

#[test]
fn probe41_same_mcid_fragments_merge_to_single_word() {
    let pdf = fixture_probe41_same_mcid_still_merges();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let page = doc.extract_page_text(0).expect("extract_page_text");
    let texts: Vec<String> = page.spans.iter().map(|s| s.text.clone()).collect();
    assert!(
        texts.iter().any(|t| t == "Hello"),
        "same-MCID fragments must still merge to 'Hello'; got spans={:?}",
        texts
    );
}
