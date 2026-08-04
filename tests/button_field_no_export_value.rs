//! A button field's /V names its selected appearance state; only /Off means
//! "unselected" (ISO 32000-1 §12.7.4.2.3). An export value of /No is a real
//! on-state — collapsing it to `Boolean(false)` makes a selected "No" answer
//! indistinguishable from an untouched field.

use pdf_oxide::extractors::{FieldValue, FormExtractor};
use pdf_oxide::PdfDocument;

fn form_pdf(button_value: &str) -> Vec<u8> {
    let mut bodies: Vec<Vec<u8>> = vec![Vec::new(); 8]; // ids 1..=7
    bodies[1] = b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [4 0 R 7 0 R] >> >>".to_vec();
    bodies[2] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec();
    bodies[3] =
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Annots [4 0 R 7 0 R] >>".to_vec();
    bodies[4] = format!(
        "<< /Type /Annot /Subtype /Widget /FT /Btn /T (Answer) /V /{button_value} /AS /{button_value} \
         /Rect [10 10 30 30] /AP << /N << /No 5 0 R /Off 6 0 R >> >> >>"
    )
    .into_bytes();
    bodies[5] = b"<< /Length 0 >>\nstream\n\nendstream".to_vec();
    bodies[6] = b"<< /Length 0 >>\nstream\n\nendstream".to_vec();
    bodies[7] =
        b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (Name) /V (Alice) /Rect [10 40 100 60] >>"
            .to_vec();

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    let mut offsets = [0usize; 8];
    for id in 1..=7 {
        offsets[id] = out.len();
        out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        out.extend_from_slice(&bodies[id]);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(b"xref\n0 8\n0000000000 65535 f \n");
    for id in 1..=7 {
        out.extend_from_slice(format!("{:010} 00000 n \n", offsets[id]).as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    out
}

fn field_value(pdf: Vec<u8>, name: &str) -> FieldValue {
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let fields = FormExtractor::extract_fields(&doc).expect("fields");
    fields
        .iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("no field named {name:?}"))
        .value
        .clone()
}

#[test]
fn selected_no_state_keeps_its_export_value() {
    let value = field_value(form_pdf("No"), "Answer");
    assert_eq!(
        value,
        FieldValue::Name("No".to_string()),
        "a selected /No on-state must keep its export value, not collapse to Boolean(false)"
    );
}

#[test]
fn off_state_still_reports_unselected() {
    let value = field_value(form_pdf("Off"), "Answer");
    assert_eq!(
        value,
        FieldValue::Boolean(false),
        "/Off is the one name that genuinely means unselected"
    );
}

#[test]
fn text_field_value_unaffected() {
    let value = field_value(form_pdf("No"), "Name");
    assert_eq!(value, FieldValue::Text("Alice".to_string()));
}
