//! Regression: filling a value on a **merged field+widget** AcroForm field
//! (the field dictionary IS the widget annotation — the common single-widget
//! case, ISO 32000-1 §12.7.4.1) must update that object in place, not mint a
//! bare orphan field object and blank the widget.
//!
//! Reproduces pdf_oxide_api#1: a `/v1/forms/fill` with a CJK value produced an
//! output where the widget the reader displays had `/V <FEFF>` (empty) while the
//! real value was stranded on a new `<< /FT /Tx /V <…> >>` object with no `/T`,
//! `/Subtype`, or `/Rect`. PyMuPDF then read the field value as empty.
//!
//! Fully synthetic fixture (no committed binary) — built in-memory with a
//! correct xref so it round-trips through `from_bytes` → fill → `save_to_bytes`.

use pdf_oxide::editor::form_fields::FormFieldValue;
use pdf_oxide::editor::DocumentEditor;
use pdf_oxide::extractors::forms::{FieldValue, FormExtractor};
use pdf_oxide::PdfDocument;

/// Build a 1-page PDF with a single merged field+widget text field named
/// `full_name`. The same object (4 0 R) is referenced by both the AcroForm
/// `/Fields` and the page `/Annots`.
fn merged_widget_form() -> Vec<u8> {
    let objs: [&str; 5] = [
        "<< /Type /Catalog /Pages 2 0 R /AcroForm 5 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Annots [4 0 R] >>",
        // Merged field+widget: field keys (/FT /T) AND annotation keys
        // (/Subtype /Widget /Rect /P) on one object.
        "<< /Type /Annot /Subtype /Widget /FT /Tx /T (full_name) \
            /Rect [50 100 250 130] /P 3 0 R /DA (/Helv 12 Tf 0 g) >>",
        "<< /Fields [4 0 R] /DA (/Helv 12 Tf 0 g) \
            /DR << /Font << /Helv << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>",
    ];

    let mut buf = String::from("%PDF-1.7\n");
    let mut offsets = Vec::with_capacity(objs.len());
    for (i, body) in objs.iter().enumerate() {
        offsets.push(buf.len());
        buf.push_str(&format!("{} 0 obj\n{}\nendobj\n", i + 1, body));
    }
    let xref_off = buf.len();
    buf.push_str(&format!("xref\n0 {}\n", objs.len() + 1));
    buf.push_str("0000000000 65535 f \n");
    for off in &offsets {
        buf.push_str(&format!("{:010} 00000 n \n", off));
    }
    buf.push_str(&format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        objs.len() + 1,
        xref_off
    ));
    buf.into_bytes()
}

#[test]
fn fill_merged_field_widget_persists_value_and_name() {
    let mut editor = DocumentEditor::from_bytes(merged_widget_form()).expect("parse fixture");
    editor
        .set_form_field_value("full_name", FormFieldValue::Text("山田太郎".to_string()))
        .expect("set value");
    let out = editor.save_to_bytes().expect("save");

    // Re-read: a single field must carry BOTH the name and the value. The bug
    // decoupled them (value on a nameless orphan, widget blanked), so no field
    // had both.
    let doc = PdfDocument::from_bytes(out).expect("reparse output");
    let fields = FormExtractor::extract_fields(&doc).expect("extract fields");

    let full_name = fields
        .iter()
        .find(|f| f.name == "full_name" || f.full_name == "full_name")
        .unwrap_or_else(|| {
            panic!(
                "field 'full_name' missing after fill+save; got {:?}",
                fields.iter().map(|f| &f.name).collect::<Vec<_>>()
            )
        });

    match &full_name.value {
        FieldValue::Text(s) => {
            assert_eq!(s, "山田太郎", "filled CJK value must persist on the named field")
        },
        other => panic!("full_name value must be Text(山田太郎), got {:?}", other),
    }
}
