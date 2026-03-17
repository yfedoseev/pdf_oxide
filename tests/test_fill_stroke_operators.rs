//! Test for B, B*, b* operators (fill-and-stroke)

use pdf_oxide::content::operators::Operator;
use pdf_oxide::content::parser::parse_content_stream;

#[test]
fn test_parse_fill_stroke_operators() {
    // Test B operator (fill and stroke, non-zero winding)
    let content = b"100 100 200 150 re B";
    let ops = parse_content_stream(content).unwrap();
    assert!(ops.iter().any(|op| matches!(op, Operator::FillStroke)), 
        "B operator should parse to FillStroke");
    
    // Test B* operator (fill and stroke, even-odd)
    let content = b"100 100 200 150 re B*";
    let ops = parse_content_stream(content).unwrap();
    assert!(ops.iter().any(|op| matches!(op, Operator::FillStrokeEvenOdd)),
        "B* operator should parse to FillStrokeEvenOdd");
    
    // Test b* operator (close, fill and stroke, even-odd)
    let content = b"100 100 m 200 100 l 200 200 l b*";
    let ops = parse_content_stream(content).unwrap();
    assert!(ops.iter().any(|op| matches!(op, Operator::CloseFillStrokeEvenOdd)),
        "b* operator should parse to CloseFillStrokeEvenOdd");
}

#[test]
fn test_existing_fill_stroke_operators() {
    // Verify existing operators still work
    
    // Test b operator (close, fill and stroke, non-zero winding)
    let content = b"100 100 m 200 100 l 200 200 l b";
    let ops = parse_content_stream(content).unwrap();
    assert!(ops.iter().any(|op| matches!(op, Operator::CloseFillStroke)),
        "b operator should parse to CloseFillStroke");
    
    // Test f operator (fill, non-zero winding)
    let content = b"100 100 200 150 re f";
    let ops = parse_content_stream(content).unwrap();
    assert!(ops.iter().any(|op| matches!(op, Operator::Fill)),
        "f operator should parse to Fill");
    
    // Test f* operator (fill, even-odd)
    let content = b"100 100 200 150 re f*";
    let ops = parse_content_stream(content).unwrap();
    assert!(ops.iter().any(|op| matches!(op, Operator::FillEvenOdd)),
        "f* operator should parse to FillEvenOdd");
    
    // Test S operator (stroke)
    let content = b"100 100 200 150 re S";
    let ops = parse_content_stream(content).unwrap();
    assert!(ops.iter().any(|op| matches!(op, Operator::Stroke)),
        "S operator should parse to Stroke");
}
