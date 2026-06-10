//! Debug aid: dump a named font resource's decode-chain-relevant structure.
//!
//! Usage: cargo run --example dump_font -- <file.pdf> [page] [FontName]
//! Resolves the page's /Resources /Font subdictionary, then prints, for the
//! requested font (default F6), the keys that drive Unicode recovery:
//! Subtype, Encoding (+Differences), ToUnicode presence, and—via
//! DescendantFonts/FontDescriptor—which embedded program (FontFile/2/3) exists.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::object::Object;

fn resolve(doc: &PdfDocument, o: &Object) -> Object {
    match o {
        Object::Reference(r) => doc.load_object(*r).unwrap_or_else(|_| o.clone()),
        _ => o.clone(),
    }
}

fn summarize(doc: &PdfDocument, o: &Object, depth: usize) {
    let pad = "  ".repeat(depth);
    let o = resolve(doc, o);
    let dict = match o.as_dict() {
        Some(d) => d.clone(),
        None => {
            println!("{pad}(not a dict: {})", o.type_name());
            return;
        },
    };
    let mut keys: Vec<_> = dict.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        let v = dict.get(k).unwrap();
        match k.as_str() {
            "Encoding" => {
                let rv = resolve(doc, v);
                if let Some(ed) = rv.as_dict() {
                    let diffs = ed.get("Differences").is_some();
                    let base = ed.get("BaseEncoding").and_then(|b| b.as_name());
                    println!("{pad}/Encoding = <dict> BaseEncoding={base:?} Differences={diffs}");
                } else {
                    println!("{pad}/Encoding = {:?}", rv.as_name());
                }
            },
            "ToUnicode" => println!("{pad}/ToUnicode = PRESENT ({})", v.type_name()),
            "DescendantFonts" => {
                println!("{pad}/DescendantFonts:");
                let rv = resolve(doc, v);
                if let Object::Array(a) = &rv {
                    for e in a {
                        summarize(doc, e, depth + 1);
                    }
                }
            },
            "FontDescriptor" => {
                println!("{pad}/FontDescriptor:");
                summarize(doc, v, depth + 1);
            },
            "FontFile" | "FontFile2" | "FontFile3" => {
                println!("{pad}/{k} = EMBEDDED PROGRAM PRESENT");
            },
            "Subtype" | "BaseFont" | "CIDToGIDMap" => {
                println!("{pad}/{k} = {:?}", resolve(doc, v).as_name());
            },
            _ => {},
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: <file.pdf> [page] [FontName]");
    let page: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let want = std::env::args().nth(3).unwrap_or_else(|| "F6".to_string());

    let doc = PdfDocument::open(&path).expect("open pdf");
    let page_obj = doc.get_page(page).expect("page");
    let page_dict = page_obj.as_dict().expect("page dict");
    let res = resolve(&doc, page_dict.get("Resources").expect("/Resources"));
    let res = res.as_dict().expect("res dict");
    let fonts = resolve(&doc, res.get("Font").expect("/Font"));
    let fonts = fonts.as_dict().expect("font dict");

    let mut names: Vec<_> = fonts.keys().cloned().collect();
    names.sort();
    println!("page {page} fonts: {names:?}\n");
    let v = fonts
        .get(&want)
        .unwrap_or_else(|| panic!("font {want} not on page {page}"));
    println!("=== /{want} ===");
    summarize(&doc, v, 0);
}
