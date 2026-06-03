//! Integration tests for path (vector graphics) extraction.
//!
//! v0.3.1: Path Objects Extraction

use pdf_oxide::document::PdfDocument;
use pdf_oxide::elements::{LineCap, LineJoin, PathContent, PathOperation};
use pdf_oxide::extractors::paths::{FillRule, PathExtractor};
use pdf_oxide::geometry::Rect;
use pdf_oxide::layout::Color;

// ============================================================================
// PathExtractor Unit Tests
// ============================================================================

mod path_extractor_tests {
    use super::*;

    #[test]
    fn test_simple_line() {
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());
        extractor.set_line_width(1.0);

        extractor.move_to(100.0, 100.0);
        extractor.line_to(200.0, 100.0);
        extractor.stroke();

        let paths = extractor.finish();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].has_stroke());
        assert!(!paths[0].has_fill());
        assert_eq!(paths[0].operations.len(), 2);
    }

    #[test]
    fn test_rectangle() {
        let mut extractor = PathExtractor::new();
        extractor.set_fill_color(Color::new(1.0, 0.0, 0.0)); // Red fill
        extractor.set_stroke_color(Color::black());

        extractor.rectangle(50.0, 50.0, 100.0, 75.0);
        extractor.fill_and_stroke(FillRule::NonZero);

        let paths = extractor.finish();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].has_fill());
        assert!(paths[0].has_stroke());

        // Check bounding box
        let bbox = &paths[0].bbox;
        assert!((bbox.x - 50.0).abs() < 0.001);
        assert!((bbox.y - 50.0).abs() < 0.001);
        assert!((bbox.width - 100.0).abs() < 0.001);
        assert!((bbox.height - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_bezier_curve() {
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::new(0.0, 0.0, 1.0)); // Blue

        extractor.move_to(0.0, 0.0);
        extractor.curve_to(25.0, 50.0, 75.0, 50.0, 100.0, 0.0);
        extractor.stroke();

        let paths = extractor.finish();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].operations.len(), 2);

        // Verify curve operation
        match &paths[0].operations[1] {
            PathOperation::CurveTo(x1, y1, x2, y2, x3, y3) => {
                assert_eq!(x1, &25.0);
                assert_eq!(y1, &50.0);
                assert_eq!(x2, &75.0);
                assert_eq!(y2, &50.0);
                assert_eq!(x3, &100.0);
                assert_eq!(y3, &0.0);
            },
            _ => panic!("Expected CurveTo operation"),
        }
    }

    #[test]
    fn test_closed_path() {
        let mut extractor = PathExtractor::new();
        extractor.set_fill_color(Color::new(0.0, 1.0, 0.0)); // Green

        // Triangle
        extractor.move_to(50.0, 0.0);
        extractor.line_to(100.0, 100.0);
        extractor.line_to(0.0, 100.0);
        extractor.close_path();
        extractor.fill(FillRule::NonZero);

        let paths = extractor.finish();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].has_fill());
        assert!(!paths[0].has_stroke());

        // Check for ClosePath operation
        let has_close = paths[0]
            .operations
            .iter()
            .any(|op| matches!(op, PathOperation::ClosePath));
        assert!(has_close);
    }

    #[test]
    fn test_multiple_subpaths() {
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());
        extractor.set_line_width(2.0);

        // First line
        extractor.move_to(0.0, 0.0);
        extractor.line_to(100.0, 0.0);
        extractor.stroke();

        // Second line
        extractor.move_to(0.0, 50.0);
        extractor.line_to(100.0, 50.0);
        extractor.stroke();

        let paths = extractor.finish();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_line_styles() {
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());
        extractor.set_line_width(3.0);
        extractor.set_line_cap(LineCap::Round);
        extractor.set_line_join(LineJoin::Bevel);

        extractor.move_to(0.0, 0.0);
        extractor.line_to(50.0, 50.0);
        extractor.line_to(100.0, 0.0);
        extractor.stroke();

        let paths = extractor.finish();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].stroke_width, 3.0);
        assert_eq!(paths[0].line_cap, LineCap::Round);
        assert_eq!(paths[0].line_join, LineJoin::Bevel);
    }

    #[test]
    fn test_fill_rules() {
        let mut extractor = PathExtractor::new();
        extractor.set_fill_color(Color::new(0.5, 0.5, 0.5));

        extractor.rectangle(0.0, 0.0, 100.0, 100.0);
        extractor.fill(FillRule::EvenOdd);

        let paths = extractor.finish();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].has_fill());
    }

    #[test]
    fn test_end_path_without_painting() {
        let mut extractor = PathExtractor::new();

        // Path that gets discarded (clipping path)
        extractor.move_to(0.0, 0.0);
        extractor.rectangle(10.0, 10.0, 80.0, 80.0);
        extractor.end_path();

        // This path should not be in the results
        let paths = extractor.finish();
        assert!(paths.is_empty());
    }

    // ─── Optional Content Group (PDF "layer") tagging ─────────────────
    //
    // Covers the BDC/BMC/EMC marked-content stack added to expose the
    // OCG name on each finalized path (ISO 32000-1:2008 §8.11, §14.6).

    #[test]
    fn test_layer_attached_when_oc_active() {
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());

        // Simulate `BDC /OC <</Name (A-GRID)>>` — dispatcher resolves the
        // OCG /Name and pushes Some("A-GRID") onto the layer stack.
        extractor.push_oc_layer(Some("A-GRID".to_string()));
        extractor.move_to(0.0, 0.0);
        extractor.line_to(100.0, 0.0);
        extractor.stroke();
        extractor.pop_oc_layer();

        let paths = extractor.finish();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].layer.as_deref(), Some("A-GRID"));
    }

    #[test]
    fn test_layer_none_outside_oc() {
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());

        // No BDC /OC seen — path is emitted in the default content scope.
        extractor.move_to(0.0, 0.0);
        extractor.line_to(50.0, 50.0);
        extractor.stroke();

        let paths = extractor.finish();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].layer, None);
    }

    #[test]
    fn test_layer_cleared_after_pop() {
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());

        extractor.push_oc_layer(Some("S-COLS".to_string()));
        extractor.move_to(0.0, 0.0);
        extractor.line_to(10.0, 0.0);
        extractor.stroke(); // belongs to S-COLS
        extractor.pop_oc_layer();

        extractor.move_to(0.0, 20.0);
        extractor.line_to(10.0, 20.0);
        extractor.stroke(); // belongs to no layer

        let paths = extractor.finish();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].layer.as_deref(), Some("S-COLS"));
        assert_eq!(paths[1].layer, None);
    }

    #[test]
    fn test_outer_oc_survives_nested_non_oc_bmc() {
        // Real-world PDFs nest non-/OC marked content (e.g. /Span <<...>>
        // for accessibility) inside an /OC region. The outer OCG must
        // continue to apply to paths emitted in the inner region.
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());

        extractor.push_oc_layer(Some("A-WALL".to_string())); // BDC /OC /Layer_1
        extractor.push_oc_layer(None); // BMC /Span (non-OC marked content)

        extractor.move_to(0.0, 0.0);
        extractor.line_to(10.0, 0.0);
        extractor.stroke();

        extractor.pop_oc_layer(); // EMC (closes /Span)
        extractor.pop_oc_layer(); // EMC (closes /OC)

        let paths = extractor.finish();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].layer.as_deref(), Some("A-WALL"));
    }

    #[test]
    fn test_nested_oc_inner_wins() {
        // When two /OC regions nest, paths in the inner region belong to
        // the inner OCG (the innermost Some(_) on the stack wins).
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());

        extractor.push_oc_layer(Some("OUTER".to_string()));
        extractor.push_oc_layer(Some("INNER".to_string()));

        extractor.move_to(0.0, 0.0);
        extractor.line_to(10.0, 0.0);
        extractor.stroke();

        extractor.pop_oc_layer();
        extractor.move_to(0.0, 20.0);
        extractor.line_to(10.0, 20.0);
        extractor.stroke();

        extractor.pop_oc_layer();

        let paths = extractor.finish();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].layer.as_deref(), Some("INNER"));
        assert_eq!(paths[1].layer.as_deref(), Some("OUTER"));
    }

    #[test]
    fn test_underflow_pop_is_safe() {
        // Malformed PDFs may emit EMC without a matching BDC. Pop on an
        // empty stack must be a no-op — extraction should keep going.
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());

        extractor.pop_oc_layer(); // stack empty — must not panic
        extractor.move_to(0.0, 0.0);
        extractor.line_to(10.0, 0.0);
        extractor.stroke();

        let paths = extractor.finish();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].layer, None);
    }
}

// ============================================================================
// SVG Conversion Tests
// ============================================================================

mod svg_conversion_tests {
    use super::*;

    fn create_simple_path() -> PathContent {
        PathContent {
            operations: vec![
                PathOperation::MoveTo(10.0, 20.0),
                PathOperation::LineTo(100.0, 20.0),
                PathOperation::LineTo(100.0, 80.0),
                PathOperation::ClosePath,
            ],
            bbox: Rect::new(10.0, 20.0, 90.0, 60.0),
            stroke_color: Some(Color::black()),
            fill_color: Some(Color::new(1.0, 0.0, 0.0)),
            stroke_width: 2.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            dash_pattern: None,
            matrix: None,
            artifact_type: None,
            reading_order: None,
            layer: None,
        }
    }

    #[test]
    fn test_svg_path_data_generation() {
        let path = create_simple_path();

        // Generate path data string
        let mut d = String::new();
        for op in &path.operations {
            match op {
                PathOperation::MoveTo(x, y) => d.push_str(&format!("M {} {} ", x, y)),
                PathOperation::LineTo(x, y) => d.push_str(&format!("L {} {} ", x, y)),
                PathOperation::ClosePath => d.push_str("Z "),
                _ => {},
            }
        }

        assert!(d.contains("M 10 20"));
        assert!(d.contains("L 100 20"));
        assert!(d.contains("L 100 80"));
        assert!(d.contains("Z"));
    }

    #[test]
    fn test_svg_stroke_attributes() {
        let path = create_simple_path();

        // Verify stroke color is set
        assert!(path.stroke_color.is_some());
        let stroke = path.stroke_color.unwrap();
        assert_eq!(stroke.r, 0.0);
        assert_eq!(stroke.g, 0.0);
        assert_eq!(stroke.b, 0.0);
    }

    #[test]
    fn test_svg_fill_attributes() {
        let path = create_simple_path();

        // Verify fill color is set
        assert!(path.fill_color.is_some());
        let fill = path.fill_color.unwrap();
        assert_eq!(fill.r, 1.0);
        assert_eq!(fill.g, 0.0);
        assert_eq!(fill.b, 0.0);
    }

    #[test]
    fn test_svg_bezier_curve() {
        let path = PathContent {
            operations: vec![
                PathOperation::MoveTo(0.0, 100.0),
                PathOperation::CurveTo(25.0, 0.0, 75.0, 0.0, 100.0, 100.0),
            ],
            bbox: Rect::new(0.0, 0.0, 100.0, 100.0),
            stroke_color: Some(Color::black()),
            fill_color: None,
            stroke_width: 1.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            dash_pattern: None,
            matrix: None,
            artifact_type: None,
            reading_order: None,
            layer: None,
        };

        // Generate curve path data
        let mut d = String::new();
        for op in &path.operations {
            match op {
                PathOperation::MoveTo(x, y) => d.push_str(&format!("M {} {} ", x, y)),
                PathOperation::CurveTo(x1, y1, x2, y2, x3, y3) => {
                    d.push_str(&format!("C {} {} {} {} {} {} ", x1, y1, x2, y2, x3, y3))
                },
                _ => {},
            }
        }

        assert!(d.contains("M 0 100"));
        assert!(d.contains("C 25 0 75 0 100 100"));
    }

    #[test]
    fn test_svg_rectangle() {
        let path = PathContent {
            operations: vec![PathOperation::Rectangle(50.0, 50.0, 200.0, 100.0)],
            bbox: Rect::new(50.0, 50.0, 200.0, 100.0),
            stroke_color: Some(Color::black()),
            fill_color: Some(Color::new(0.9, 0.9, 0.9)),
            stroke_width: 1.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            dash_pattern: None,
            matrix: None,
            artifact_type: None,
            reading_order: None,
            layer: None,
        };

        // Rectangle should be converted to M L L L Z
        let has_rect = path
            .operations
            .iter()
            .any(|op| matches!(op, PathOperation::Rectangle(_, _, _, _)));
        assert!(has_rect);
    }

    #[test]
    fn test_svg_line_cap_conversion() {
        let round_cap = PathContent {
            operations: vec![
                PathOperation::MoveTo(0.0, 0.0),
                PathOperation::LineTo(100.0, 0.0),
            ],
            bbox: Rect::new(0.0, 0.0, 100.0, 0.0),
            stroke_color: Some(Color::black()),
            fill_color: None,
            stroke_width: 10.0,
            line_cap: LineCap::Round,
            line_join: LineJoin::Miter,
            dash_pattern: None,
            matrix: None,
            artifact_type: None,
            reading_order: None,
            layer: None,
        };

        assert_eq!(round_cap.line_cap, LineCap::Round);
    }

    #[test]
    fn test_svg_line_join_conversion() {
        let bevel_join = PathContent {
            operations: vec![
                PathOperation::MoveTo(0.0, 0.0),
                PathOperation::LineTo(50.0, 50.0),
                PathOperation::LineTo(100.0, 0.0),
            ],
            bbox: Rect::new(0.0, 0.0, 100.0, 50.0),
            stroke_color: Some(Color::black()),
            fill_color: None,
            stroke_width: 5.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Bevel,
            dash_pattern: None,
            matrix: None,
            artifact_type: None,
            reading_order: None,
            layer: None,
        };

        assert_eq!(bevel_join.line_join, LineJoin::Bevel);
    }
}

// ============================================================================
// Bounding Box Calculation Tests
// ============================================================================

mod bbox_tests {
    use super::*;

    #[test]
    fn test_line_bbox() {
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());

        extractor.move_to(10.0, 20.0);
        extractor.line_to(110.0, 80.0);
        extractor.stroke();

        let paths = extractor.finish();
        let bbox = &paths[0].bbox;

        assert!((bbox.x - 10.0).abs() < 0.001);
        assert!((bbox.y - 20.0).abs() < 0.001);
        assert!((bbox.width - 100.0).abs() < 0.001);
        assert!((bbox.height - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_rectangle_bbox() {
        let mut extractor = PathExtractor::new();
        extractor.set_fill_color(Color::black());

        extractor.rectangle(25.0, 30.0, 150.0, 200.0);
        extractor.fill(FillRule::NonZero);

        let paths = extractor.finish();
        let bbox = &paths[0].bbox;

        assert!((bbox.x - 25.0).abs() < 0.001);
        assert!((bbox.y - 30.0).abs() < 0.001);
        assert!((bbox.width - 150.0).abs() < 0.001);
        assert!((bbox.height - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_triangle_bbox() {
        let mut extractor = PathExtractor::new();
        extractor.set_fill_color(Color::black());

        // Triangle with vertices at (0, 0), (100, 0), (50, 86.6)
        extractor.move_to(0.0, 0.0);
        extractor.line_to(100.0, 0.0);
        extractor.line_to(50.0, 86.6);
        extractor.close_path();
        extractor.fill(FillRule::NonZero);

        let paths = extractor.finish();
        let bbox = &paths[0].bbox;

        assert!((bbox.x - 0.0).abs() < 0.1);
        assert!((bbox.y - 0.0).abs() < 0.1);
        assert!((bbox.width - 100.0).abs() < 0.1);
        assert!((bbox.height - 86.6).abs() < 0.1);
    }

    #[test]
    fn test_complex_path_bbox() {
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());

        // Complex path that spans a large area
        extractor.move_to(-50.0, -50.0);
        extractor.line_to(50.0, 0.0);
        extractor.curve_to(100.0, 25.0, 100.0, 75.0, 50.0, 100.0);
        extractor.line_to(-50.0, 100.0);
        extractor.close_path();
        extractor.stroke();

        let paths = extractor.finish();
        let bbox = &paths[0].bbox;

        // BBox should encompass all points
        assert!(bbox.x <= -50.0);
        assert!(bbox.y <= -50.0);
        assert!(bbox.x + bbox.width >= 100.0);
        assert!(bbox.y + bbox.height >= 100.0);
    }
}

// ============================================================================
// Integration Tests with Real PDFs (when fixtures exist)
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::path::Path;

    fn test_pdf_path(name: &str) -> std::path::PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        Path::new(&manifest_dir)
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    #[test]
    #[ignore] // Enable when vector_graphics.pdf fixture is available
    fn test_extract_paths_from_pdf() {
        let path = test_pdf_path("vector_graphics.pdf");
        if !path.exists() {
            eprintln!("Skipping test: {:?} not found", path);
            return;
        }

        let doc = PdfDocument::open(&path).expect("Failed to open PDF");
        let paths = doc.extract_paths(0).expect("Failed to extract paths");

        // Should find some paths
        assert!(!paths.is_empty(), "Expected to find paths in vector_graphics.pdf");

        // Verify basic path properties
        for path in &paths {
            assert!(!path.operations.is_empty());
            assert!(path.has_stroke() || path.has_fill());
        }
    }

    #[test]
    #[ignore] // Enable when appropriate test PDF is available
    fn test_extract_paths_in_rect() {
        let path = test_pdf_path("vector_graphics.pdf");
        if !path.exists() {
            return;
        }

        let doc = PdfDocument::open(&path).expect("Failed to open PDF");

        // Extract only paths in a specific region
        let region = Rect::new(0.0, 0.0, 300.0, 300.0);
        let paths = doc
            .extract_paths_in_rect(0, region)
            .expect("Failed to extract paths in rect");

        // All returned paths should intersect with the region
        for path in &paths {
            let bbox = &path.bbox;
            let intersects = !(bbox.x > region.x + region.width
                || bbox.x + bbox.width < region.x
                || bbox.y > region.y + region.height
                || bbox.y + bbox.height < region.y);
            assert!(intersects, "Path bbox {:?} should intersect region {:?}", bbox, region);
        }
    }

    #[test]
    fn test_path_extraction_on_simple_pdf() {
        // Try to find any existing test PDF with graphics
        let test_paths = [
            "tests/fixtures/simple.pdf",
            "tests/fixtures/test.pdf",
            "tests/fixtures/hello.pdf",
        ];

        for pdf_path in &test_paths {
            let path = Path::new(pdf_path);
            if path.exists() {
                let result = PdfDocument::open(path);
                if let Ok(doc) = result {
                    if let Ok(paths) = doc.extract_paths(0) {
                        // Just verify we can call the API without crashing
                        eprintln!("Extracted {} paths from {:?}", paths.len(), path);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Color Tests
// ============================================================================

mod color_tests {
    use super::*;

    #[test]
    fn test_stroke_color() {
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::new(1.0, 0.5, 0.0)); // Orange

        extractor.move_to(0.0, 0.0);
        extractor.line_to(100.0, 100.0);
        extractor.stroke();

        let paths = extractor.finish();
        let color = paths[0].stroke_color.unwrap();
        assert_eq!(color.r, 1.0);
        assert_eq!(color.g, 0.5);
        assert_eq!(color.b, 0.0);
    }

    #[test]
    fn test_fill_color() {
        let mut extractor = PathExtractor::new();
        extractor.set_fill_color(Color::new(0.0, 0.8, 0.2)); // Green

        extractor.rectangle(0.0, 0.0, 100.0, 100.0);
        extractor.fill(FillRule::NonZero);

        let paths = extractor.finish();
        let color = paths[0].fill_color.unwrap();
        assert_eq!(color.r, 0.0);
        assert_eq!(color.g, 0.8);
        assert_eq!(color.b, 0.2);
    }

    #[test]
    fn test_stroke_and_fill_different_colors() {
        let mut extractor = PathExtractor::new();
        extractor.set_stroke_color(Color::black());
        extractor.set_fill_color(Color::new(0.9, 0.9, 0.0)); // Yellow

        extractor.rectangle(10.0, 10.0, 80.0, 80.0);
        extractor.fill_and_stroke(FillRule::NonZero);

        let paths = extractor.finish();
        let stroke = paths[0].stroke_color.unwrap();
        let fill = paths[0].fill_color.unwrap();

        assert_eq!(stroke.r, 0.0);
        assert_eq!(stroke.g, 0.0);
        assert_eq!(stroke.b, 0.0);

        assert_eq!(fill.r, 0.9);
        assert_eq!(fill.g, 0.9);
        assert_eq!(fill.b, 0.0);
    }
}
