//! Regression tests for MAJOR-class issues uncovered alongside the
//! struct-tree-scope `/ActualText` refactor.
//!
//! MAJOR-1 — Nested BDC must restore the outer MCID on EMC, not blank it.
//! MAJOR-2 — A single `/Span << /MCID n /ActualText (s) >>` BDC scope
//!           must emit `s` ONCE no matter how many Tj operators sit
//!           inside, not once per Tj.
//!
//! These pin behaviours at end-to-end level so future refactors can't
//! regress them silently.

use pdf_oxide::document::PdfDocument;

// Carbon-copy of the test PdfBuilder from
// `test_struct_actualtext_critical.rs`. Integration crates can't share
// modules so we re-declare the helpers we need.

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
    fn page(mut self, p: u32) -> Self {
        self.page_obj = Some(p);
        self
    }
    #[allow(dead_code)]
    fn actual_text(mut self, t: &str) -> Self {
        self.actual_text = Some(t.to_string());
        self
    }
    fn k(mut self, c: K) -> Self {
        self.children.push(c);
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
    fn add_page_content(&mut self, c: Vec<u8>) -> usize {
        self.page_contents.push(c);
        self.parent_tree_entries.push(Vec::new());
        self.page_contents.len() - 1
    }
    fn add_elem(&mut self, e: Elem) -> u32 {
        let n = e.obj_num;
        self.elems.push(e);
        n
    }
    fn register_mcid(&mut self, page_idx: usize, mcid: u32, elem_obj: u32) {
        self.parent_tree_entries[page_idx].push((mcid, elem_obj));
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
        use std::collections::BTreeMap;
        let n_pages = self.page_contents.len() as u32;
        let catalog = 1u32;
        let pages = 2u32;
        let font = 3u32;
        let first_page = 4u32;
        let first_content = first_page + n_pages;
        let parent_tree = first_content + n_pages;
        let struct_tree_root = parent_tree + 1;
        let first_struct_elem = struct_tree_root + 1;

        let mut used = std::collections::HashSet::new();
        for e in &self.elems {
            assert!(e.obj_num >= first_struct_elem);
            assert!(used.insert(e.obj_num));
        }
        let root_elem = self.elems[0].obj_num;

        let mut objs: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
        objs.insert(
            catalog,
            format!(
                "<< /Type /Catalog /Pages {} 0 R /MarkInfo << /Marked true >> /StructTreeRoot {} 0 R >>",
                pages, struct_tree_root
            )
            .into_bytes(),
        );
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
            let mut stream =
                format!("<< /Length {} >>\nstream\n", content_bytes.len()).into_bytes();
            stream.extend_from_slice(content_bytes);
            stream.extend_from_slice(b"\nendstream");
            objs.insert(content_obj, stream);
            objs.insert(
                first_page + i,
                format!(
                    "<< /Type /Page /Parent {} 0 R /MediaBox [0 0 612 792] \
                     /Resources << /Font << /F1 {} 0 R >> /ProcSet [/PDF /Text] >> \
                     /Contents {} 0 R /StructParents {} >>",
                    pages, font, content_obj, i
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

// =============================================================
// MAJOR-1: nested BDC inside one MCID-bearing BDC; on inner EMC the
// outer MCID must be restored, not blanked. The third Tj below sits
// directly under MCID 10 — not under MCID 11 (popped) — and must be
// attributed to MCID 10.
// =============================================================
//
// Page content:
//   BDC MCID 10
//     (A) Tj
//     BDC MCID 11
//       (B) Tj
//     EMC                       ← restore current_mcid → 10
//     (C) Tj                    ← must be MCID 10
//   EMC
//
// Structure:
//   P /K MCID 10  (spans for "A" and "C")
//   Span /K MCID 11 (span for "B")
//   Span carries /ActualText "[B-replacement]" — when MAJOR-1 is fixed
//   AND the new ActualText path works correctly the output is
//   "A [B-replacement] C", in that order.
//
// When MAJOR-1 is broken: the third Tj has mcid=None, so it appears
// neither under MCID 10's structure node nor under MCID 11's — the
// glyph "C" lands in the "unconsumed MCIDs" tail OR vanishes. The
// minimal repro asserts the structure-tree order positions "C"
// immediately after the inner B replacement.

fn fixture_major1_nested_bdc() -> Vec<u8> {
    // Page content stream:
    //   BDC MCID 10  → P-A
    //     (A) Tj
    //     BDC MCID 11 → inner Span
    //       (B) Tj
    //     EMC                  ← MCID stack must restore 10 here
    //     (C) Tj               ← THIS must attribute to MCID 10, not None
    //   EMC
    //   BDC MCID 12 → P-D
    //     (D) Tj
    //   EMC
    //
    // Structure tree gives reading order: P-A (MCID 10), inner (11),
    // P-D (MCID 12). If C ends up with mcid=None (MAJOR-1 bug),
    // structure-order assembler appends it AFTER all known MCIDs,
    // producing "A B D C" instead of "A B C D".
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
        /Span << /MCID 10 >> BDC\n\
          (A) Tj\n\
          /Span << /MCID 11 >> BDC\n\
            (B) Tj\n\
          EMC\n\
          (C) Tj\n\
        EMC\n\
        0 -20 Td\n\
        /Span << /MCID 12 >> BDC\n\
          (D) Tj\n\
        EMC\nET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());

    // Tree: Document → P-A (10) → inner Span (11); Document → P-D (12)
    // Structure-tree pre-order MCID list: 10, 11, 12.
    let _doc = b.add_elem(
        Elem::new(8, "Document", 7)
            .page(4)
            .k(K::Obj(9))
            .k(K::Obj(11)),
    );
    let _p_a = b.add_elem(Elem::new(9, "P", 8).page(4).k(K::Mcid(10, 0)).k(K::Obj(10)));
    let _inner = b.add_elem(Elem::new(10, "Span", 9).page(4).k(K::Mcid(11, 0)));
    let _p_d = b.add_elem(Elem::new(11, "P", 8).page(4).k(K::Mcid(12, 0)));
    b.register_mcid(0, 10, 9); // outer MCID 10 → P-A
    b.register_mcid(0, 11, 10); // inner MCID 11 → Span
    b.register_mcid(0, 12, 11); // sibling MCID 12 → P-D
    b.build()
}

#[test]
fn major1_nested_bdc_restores_outer_mcid() {
    let pdf = fixture_major1_nested_bdc();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    // All glyphs must appear.
    for g in ['A', 'B', 'C', 'D'] {
        assert!(extracted.contains(g), "{}: glyph must appear in output, got {:?}", g, extracted);
    }
    // C must precede D — the discriminator for MAJOR-1. When MAJOR-1
    // is broken, C carries mcid=None and lands in the "unconsumed
    // MCIDs" tail of the structure-order assembler, AFTER MCID 12's
    // span "D". When MAJOR-1 is fixed, C carries the outer MCID 10
    // and stays inside P-A's group, which precedes P-D in the tree.
    let c_pos = extracted.find('C').unwrap();
    let d_pos = extracted.find('D').unwrap();
    assert!(
        c_pos < d_pos,
        "C must precede D — MAJOR-1: EMC of inner BDC must restore outer MCID. \
         Got positions C={} D={} in {:?}",
        c_pos,
        d_pos,
        extracted
    );
}

// =============================================================
// MAJOR-2: one BDC scope with /ActualText must emit ONCE no matter
// how many Tj operators it contains.
// =============================================================
//
// Page content:
//   BDC MCID 0 /ActualText "mc"
//     (X) Tj
//     (Y) Tj
//   EMC

fn fixture_major2_multi_tj_one_scope() -> Vec<u8> {
    let content = b"BT\n/F1 12 Tf\n50 700 Td\n\
        /Span << /MCID 0 /ActualText <FEFF006D0063> >> BDC\n\
        (X) Tj\n\
        (Y) Tj\n\
        EMC\nET\n";
    let mut b = PdfBuilder::new();
    b.add_page_content(content.to_vec());
    let _ = b.add_elem(Elem::new(8, "Span", 7).page(4).k(K::Mcid(0, 0)));
    b.register_mcid(0, 0, 8);
    b.build()
}

#[test]
fn major2_mc_scope_actualtext_emits_once_per_scope() {
    let pdf = fixture_major2_multi_tj_one_scope();
    let doc = PdfDocument::from_bytes(pdf).expect("open");
    let extracted = doc.extract_text(0).expect("extract_text");
    let occurrences = extracted.matches("mc").count();
    assert_eq!(
        occurrences, 1,
        "BDC /ActualText 'mc' replaces the entire MC-scope sequence: one emission, not one per Tj. Got {:?}",
        extracted
    );
    assert!(!extracted.contains('X'), "raw 'X' must NOT leak, got {:?}", extracted);
    assert!(!extracted.contains('Y'), "raw 'Y' must NOT leak, got {:?}", extracted);
}
