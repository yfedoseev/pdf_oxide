# Markdown Converter Usage Guide

## Quick Start

### Basic Usage

```rust
use pdf_oxide::pipeline::{OrderedTextSpan, TextPipelineConfig};
use pdf_oxide::pipeline::converters::{MarkdownOutputConverter, OutputConverter};

let converter = MarkdownOutputConverter::new();
let config = TextPipelineConfig::default();
let markdown = converter.convert(&spans, &config)?;
println!("{}", markdown);
```

### With Features Enabled

```rust
use pdf_oxide::pipeline::config::{OutputConfig, BoldMarkerBehavior};

let config = TextPipelineConfig {
    output: OutputConfig {
        detect_headings: true,
        bold_marker_behavior: BoldMarkerBehavior::Aggressive,
        extract_tables: true,
        preserve_layout: true,
        include_images: true,
        image_output_dir: Some("/tmp/images".to_string()),
    },
    ..Default::default()
};

let markdown = converter.convert(&spans, &config)?;
```

## Configuration Options

### `OutputConfig` Fields

#### Heading Detection
```rust
pub detect_headings: bool
```

**Default**: `false`

Enables automatic heading level detection based on font size:
- H1: 24pt+
- H2: 18-23pt
- H3: 14-17pt

**Example**:
```rust
OutputConfig {
    detect_headings: true,
    ..Default::default()
}
```

---

#### Bold Marker Behavior
```rust
pub bold_marker_behavior: BoldMarkerBehavior
```

**Type**: `enum BoldMarkerBehavior`
- `Conservative`: Apply bold only to text with content (skip whitespace)
- `Aggressive`: Apply bold to all bold-weight text

**Default**: `Conservative`

**Example**:
```rust
OutputConfig {
    bold_marker_behavior: BoldMarkerBehavior::Aggressive,
    ..Default::default()
}
```

---

#### Table Extraction
```rust
pub extract_tables: bool
```

**Default**: `false`

Detects grid-aligned text and formats as markdown tables:

**Example Input**:
```
Name    Age    City
Alice   30     NYC
Bob     25     LA
```

**Example Output**:
```markdown
| Name  | Age | City |
|-------|-----|------|
| Alice | 30  | NYC  |
| Bob   | 25  | LA   |
```

---

#### Layout Preservation
```rust
pub preserve_layout: bool
```

**Default**: `false`

Preserves whitespace to maintain column alignment:

**With `false`** (normalized):
```
Text in column A    Text in column B
More text
```

**With `true`** (preserved):
```
Text in column A                Text in column B
More text
```

---

#### Image Embedding
```rust
pub include_images: bool
pub image_output_dir: Option<String>
```

**Default**: `include_images: true`, `image_output_dir: None`

Controls image handling in output:
- `include_images: true`: Include image references
- `image_output_dir: Some(path)`: Extract images to specified directory

**Example**:
```rust
OutputConfig {
    include_images: true,
    image_output_dir: Some("/home/user/pdf_images".to_string()),
    ..Default::default()
}
```

---

## Feature Examples

### Example 1: Academic Paper Conversion

```rust
let config = TextPipelineConfig {
    output: OutputConfig {
        detect_headings: true,
        bold_marker_behavior: BoldMarkerBehavior::Conservative,
        extract_tables: true,
        preserve_layout: false,
        ..Default::default()
    },
    ..Default::default()
};

// Result:
// # Introduction
//
// This paper discusses **important** concepts.
//
// | Metric | Value |
// |--------|-------|
// | Recall | 95%   |
// | F1     | 0.92  |
```

### Example 2: Form Extraction

```rust
let config = TextPipelineConfig {
    output: OutputConfig {
        preserve_layout: true,
        bold_marker_behavior: BoldMarkerBehavior::Conservative,
        ..Default::default()
    },
    ..Default::default()
};

// Result:
// Name: ________________    Date: __________
// Address: ___________________________________
```

### Example 3: Simple Text Conversion

```rust
let config = TextPipelineConfig::default();

// Result:
// Plain text with **bold** and *italic* preserved
// but no heading detection or tables
```

## Output Formatting

### Headings
```markdown
# H1 Title
## H2 Section
### H3 Subsection
```

### Text Formatting
```markdown
**bold text**
*italic text*
***bold italic***
```

### Tables
```markdown
| Column 1 | Column 2 |
|----------|----------|
| Cell A   | Cell B   |
| Cell C   | Cell D   |
```

### Links
Auto-converted from text:
```markdown
[https://example.com](https://example.com)
[user@example.com](mailto:user@example.com)
```

### Paragraphs
Separated by blank lines:
```markdown
First paragraph.

Second paragraph.
```

## Performance Considerations

### Recommended Settings for Different Use Cases

**Speed-Optimized** (fastest):
```rust
OutputConfig {
    detect_headings: false,
    extract_tables: false,
    preserve_layout: false,
    ..Default::default()
}
```

**Quality-Optimized** (best output):
```rust
OutputConfig {
    detect_headings: true,
    extract_tables: true,
    preserve_layout: false,
    bold_marker_behavior: BoldMarkerBehavior::Conservative,
    ..Default::default()
}
```

**Form-Optimized** (preserves layout):
```rust
OutputConfig {
    preserve_layout: true,
    detect_headings: false,
    extract_tables: false,
    ..Default::default()
}
```

### Time Complexity

- **Overall**: O(n) where n = number of text spans
- **Heading detection**: O(n) - single statistical pass
- **Table detection**: O(n) - grouping by position
- **Linkification**: O(n) - regex replacements
- **Formatting**: O(n) - inline formatting

No heap allocations in tight loops.

## Testing

### Running Tests

```bash
# Run all markdown converter tests
cargo test --test test_pipeline_markdown_converter

# Run specific test
cargo test --test test_pipeline_markdown_converter test_heading_detection_h1

# Run with output
cargo test --test test_pipeline_markdown_converter -- --nocapture
```

### Writing Custom Tests

```rust
#[test]
fn my_custom_test() {
    use pdf_oxide::layout::{Color, FontWeight, TextSpan};
    use pdf_oxide::pipeline::OrderedTextSpan;

    let span = OrderedTextSpan::new(
        TextSpan {
            text: "My text".to_string(),
            bbox: Rect::new(0.0, 100.0, 100.0, 24.0),
            font_name: "Arial".to_string(),
            font_size: 24.0,
            font_weight: FontWeight::Bold,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: 0,
            offset_semantic: false,
            split_boundary_before: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
        },
        0,
    );

    let converter = MarkdownOutputConverter::new();
    let config = TextPipelineConfig {
        output: OutputConfig {
            detect_headings: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let output = converter.convert(&vec![span], &config).unwrap();
    assert!(output.contains("# My text"));
}
```

## Troubleshooting

### Issue: Headings not detected

**Cause**: Font size below 24pt threshold

**Solution**:
1. Check actual font sizes in PDF
2. Use ratio-based detection (supply range of sizes)
3. Consider using `preserve_layout` instead

### Issue: False positive tables

**Cause**: Any grid-aligned text detected as table

**Solution**: Set `extract_tables: false`

### Issue: Whitespace not preserved

**Cause**: Layout preservation disabled

**Solution**: Enable `preserve_layout: true`

### Issue: Bold/italic not formatting

**Cause**: Font weight or italic flag not set in TextSpan

**Solution**:
1. Verify PDF contains font styling info
2. Check font mapping is correct
3. Use Conservative bold behavior

## Advanced: Custom Converter

To create a custom markdown converter:

```rust
use pdf_oxide::pipeline::converters::OutputConverter;

struct MyCustomMarkdown;

impl OutputConverter for MyCustomMarkdown {
    fn convert(&self, spans: &[OrderedTextSpan], config: &TextPipelineConfig) -> Result<String> {
        // Your implementation
        Ok("...".to_string())
    }

    fn name(&self) -> &'static str {
        "MyCustomMarkdown"
    }

    fn mime_type(&self) -> &'static str {
        "text/markdown"
    }
}
```

## See Also

- [PDF Specification](../spec/pdf.md) - ISO 32000-1:2008
- [Architecture](ARCHITECTURE.md) - System design
- [Development Guide](DEVELOPMENT_GUIDE.md) - Development workflow
