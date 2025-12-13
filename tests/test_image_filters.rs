#[cfg(feature = "ocr")]
mod image_filter_tests {
    use pdf_oxide::document::PdfDocument;

    #[test]
    fn test_image_filters_in_pride_prejudice() {
        let pdf_path = "scanned_samples/pride_prejudice.pdf";
        if !std::path::Path::new(pdf_path).exists() {
            println!("PDF not found, skipping test");
            return;
        }

        let mut doc = match PdfDocument::open(pdf_path) {
            Ok(d) => d,
            Err(e) => {
                println!("Failed to open PDF: {}", e);
                return;
            }
        };

        match doc.get_page(1) {
            Ok(page) => {
                println!("\n=== Image Filter Analysis ===");

                match page.get_resources() {
                    Ok(resources) => {
                        if let Some(xobjects) = resources.get("XObject") {
                            match xobjects.as_dict() {
                                Ok(xobj_dict) => {
                                    for (name, xobj_ref) in xobj_dict.iter().take(5) {
                                        println!("\nXObject: {}", name);

                                        match doc.resolve_indirect(xobj_ref) {
                                            Ok(xobj) => {
                                                match xobj.as_dict() {
                                                    Ok(dict) => {
                                                        if let Some(filter) = dict.get("Filter") {
                                                            println!("  Filter: {:?}", filter);
                                                        } else {
                                                            println!("  No Filter");
                                                        }

                                                        if let Some(width) = dict.get("Width") {
                                                            println!("  Width: {:?}", width);
                                                        }
                                                        if let Some(height) = dict.get("Height") {
                                                            println!("  Height: {:?}", height);
                                                        }
                                                        if let Some(bits) = dict.get("BitsPerComponent") {
                                                            println!("  BitsPerComponent: {:?}", bits);
                                                        }
                                                    }
                                                    Err(e) => println!("  Failed to get dict: {}", e),
                                                }
                                            }
                                            Err(e) => println!("  Failed to resolve: {}", e),
                                        }
                                    }
                                }
                                Err(e) => println!("Failed to get XObject dict: {}", e),
                            }
                        }
                    }
                    Err(e) => println!("Failed to get resources: {}", e),
                }
            }
            Err(e) => println!("Failed to get page: {}", e),
        }
    }
}
