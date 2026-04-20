use pdf_oxide::document::PdfDocument;
use pdf_oxide::writer::{PageBuilder, PdfWriter};

fn put(page: &mut PageBuilder<'_>, text: &str, x: f32, y: f32, font: &str, size: f32) {
    page.add_text(text, x, y, font, size);
    // Force a BT/ET boundary so adjacent add_text calls are emitted as
    // separate text objects. Without this, the extractor may merge them
    // into a single span, making these regression assertions ineffective.
    page.draw_rect(0.0, 0.0, 0.0, 0.0);
}

fn build_and_extract(build_fn: impl FnOnce(&mut PdfWriter)) -> String {
    let mut writer = PdfWriter::new();
    build_fn(&mut writer);
    let bytes = writer.finish().expect("build PDF");
    let doc = PdfDocument::from_bytes(bytes).expect("open PDF");
    doc.extract_text(0).expect("extract page 0")
}

#[test]
fn superscript_12pt_offset_stays_on_one_line() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "8", 140.0, 180.0, "Helvetica", 28.0);
        put(&mut page, "th", 156.0, 192.0, "Helvetica", 20.0);
    });

    let out = out.trim_end();
    assert!(!out.contains('\n'), "got {:?}", out);
    assert!(out.contains('8') && out.contains("th"), "got {:?}", out);
}

#[test]
fn superscript_extracts_with_correct_glyph_order() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "8", 140.0, 180.0, "Helvetica", 28.0);
        put(&mut page, "th", 156.0, 192.0, "Helvetica", 20.0);
    });

    assert_eq!(out.trim_end(), "8th", "got {:?}", out.trim_end());
}

#[test]
fn subscript_between_baseline_letters_stays_in_reading_order() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "H", 100.0, 200.0, "Helvetica", 14.0);
        put(&mut page, "2", 112.0, 197.0, "Helvetica", 9.0);
        put(&mut page, "O", 122.0, 200.0, "Helvetica", 14.0);
    });

    let collapsed: String = out.split_whitespace().collect();
    assert_eq!(collapsed, "H2O", "got {:?}", out.trim_end());
}

#[test]
fn three_glyph_run_in_distinct_bands_is_x_ordered() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "a", 100.0, 200.0, "Helvetica", 14.0);
        put(&mut page, "b", 112.0, 203.0, "Helvetica", 12.0);
        put(&mut page, "c", 124.0, 206.0, "Helvetica", 10.0);
    });

    let collapsed: String = out.split_whitespace().collect();
    assert_eq!(collapsed, "abc", "got {:?}", out.trim_end());
}

#[test]
fn baseline_same_y_stays_on_one_line() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "8", 140.0, 180.0, "Helvetica", 28.0);
        put(&mut page, "th", 156.0, 180.0, "Helvetica", 20.0);
    });

    assert_eq!(out.trim_end(), "8th", "got {:?}", out.trim_end());
}

#[test]
fn two_lines_normal_leading_still_split() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "First line of body text.", 72.0, 700.0, "Helvetica", 12.0);
        put(&mut page, "Second line of body text.", 72.0, 685.6, "Helvetica", 12.0);
    });

    let first = out.find("First line").expect("first line present");
    let second = out.find("Second line").expect("second line present");
    let between = &out[first + "First line".len()..second];
    assert!(between.contains('\n'), "got {:?}", out);
}

#[test]
fn multi_line_body_text_preserves_breaks() {
    let lines = [
        "LineAAA", "LineBBB", "LineCCC", "LineDDD", "LineEEE", "LineFFF",
    ];

    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        let mut y = 700.0;
        for line in &lines {
            put(&mut page, line, 72.0, y, "Helvetica", 12.0);
            y -= 14.4;
        }
    });

    for pair in lines.windows(2) {
        let a = out
            .find(pair[0])
            .unwrap_or_else(|| panic!("missing {:?}: {:?}", pair[0], out));
        let b = out
            .find(pair[1])
            .unwrap_or_else(|| panic!("missing {:?}: {:?}", pair[1], out));
        let between = &out[a + pair[0].len()..b];
        assert!(
            between.contains('\n'),
            "missing newline between {:?} and {:?}: {:?}",
            pair[0],
            pair[1],
            out
        );
    }
}

#[test]
fn superscript_then_next_line_still_breaks() {
    let out = build_and_extract(|w| {
        let mut page = w.add_letter_page();
        put(&mut page, "8", 140.0, 700.0, "Helvetica", 28.0);
        put(&mut page, "th", 156.0, 712.0, "Helvetica", 20.0);
        put(&mut page, "Next paragraph line.", 72.0, 672.0, "Helvetica", 12.0);
    });

    assert!(out.contains('8') && out.contains("th"), "got {:?}", out);

    let super_end = out
        .find("th")
        .or_else(|| out.find('8'))
        .expect("superscript present");
    let body_start = out.find("Next paragraph line").expect("body line present");
    let between = &out[super_end..body_start];
    assert!(between.contains('\n'), "got {:?}", out);
}
