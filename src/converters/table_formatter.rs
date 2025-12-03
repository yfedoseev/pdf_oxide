//! Markdown table formatting for detected tables.

use crate::converters::TableFormatConfig;
use crate::layout::Table;
use crate::layout::TextBlock;
use log::{debug, info, trace};
use std::cmp::max;

/// Markdown table formatter.
///
/// Converts detected table structures to valid markdown table syntax.
pub struct MarkdownTableFormatter;

impl MarkdownTableFormatter {
    /// Format a table as markdown.
    pub fn format_table(table: &Table, blocks: &[TextBlock], config: &TableFormatConfig) -> String {
        info!("Formatting table with {} rows and {} columns", table.num_rows, table.num_cols);
        trace!("Table formatting config: {:?}", config);

        if table.num_rows == 0 || table.num_cols == 0 {
            debug!("Empty table detected, returning empty string");
            return String::new();
        }

        let cell_contents = Self::extract_cell_contents(table, blocks, config);
        debug!(
            "Extracted cell contents: {} rows × {} columns",
            cell_contents.len(),
            cell_contents.iter().map(|r| r.len()).max().unwrap_or(0)
        );

        let column_widths = Self::calculate_column_widths(&cell_contents, config);
        debug!("Calculated column widths: {:?}", column_widths);

        let mut markdown = String::new();

        if !cell_contents.is_empty() {
            markdown.push_str(&Self::format_row(&cell_contents[0], &column_widths, config));
            markdown.push('\n');

            if config.include_header_separator {
                markdown.push_str(&Self::format_separator_row(&column_widths, config));
                markdown.push('\n');
            }

            for row in &cell_contents[1..] {
                markdown.push_str(&Self::format_row(row, &column_widths, config));
                markdown.push('\n');
            }
        }

        if markdown.ends_with('\n') {
            markdown.pop();
        }

        info!("Table formatting complete: {} characters", markdown.len());
        markdown
    }

    fn extract_cell_contents(
        table: &Table,
        blocks: &[TextBlock],
        config: &TableFormatConfig,
    ) -> Vec<Vec<String>> {
        let mut result = vec![];

        for row_cells in &table.cells {
            let mut row = vec![];

            for &block_idx in row_cells {
                let content = if block_idx < blocks.len() {
                    blocks[block_idx].text.trim().to_string()
                } else {
                    String::new()
                };

                let cell_content = if content.is_empty() {
                    config.empty_cell_text.clone()
                } else if config.preserve_cell_formatting {
                    content
                } else {
                    Self::strip_formatting(&content)
                };

                trace!("Cell content: '{}' (from block {})", cell_content, block_idx);
                row.push(cell_content);
            }

            result.push(row);
        }

        result
    }

    fn strip_formatting(content: &str) -> String {
        content
            .replace("**", "")
            .replace("__", "")
            .replace("*", "")
            .replace("_", "")
    }

    fn calculate_column_widths(
        cell_contents: &[Vec<String>],
        config: &TableFormatConfig,
    ) -> Vec<usize> {
        if cell_contents.is_empty() {
            return vec![];
        }

        let num_cols = cell_contents.iter().map(|r| r.len()).max().unwrap_or(0);
        let mut widths = vec![config.min_column_width; num_cols];

        for row in cell_contents {
            for (col_idx, cell) in row.iter().enumerate() {
                if col_idx < widths.len() {
                    let required_width = cell.len() + (config.cell_padding * 2);
                    widths[col_idx] = max(widths[col_idx], required_width);
                }
            }
        }

        debug!("Column widths: {:?}", widths);
        widths
    }

    fn format_row(row: &[String], column_widths: &[usize], config: &TableFormatConfig) -> String {
        let mut result = String::from("|");
        let padding = " ".repeat(config.cell_padding);

        for (col_idx, cell) in row.iter().enumerate() {
            let width = column_widths
                .get(col_idx)
                .copied()
                .unwrap_or(config.min_column_width);
            result.push_str(&padding);
            result.push_str(&format!("{:<width$}", cell, width = width));
            result.push_str(&padding);
            result.push('|');
        }

        result
    }

    fn format_separator_row(column_widths: &[usize], config: &TableFormatConfig) -> String {
        let mut separator = String::from("|");
        let padding = " ".repeat(config.cell_padding);

        for &width in column_widths {
            let sep_width = max(width, 3);
            separator.push_str(&padding);
            separator.push_str(&"-".repeat(sep_width));
            separator.push_str(&padding);
            separator.push('|');
        }

        separator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::layout::{Color, FontWeight, TextChar};

    fn mock_block(text: &str, x: f32, y: f32) -> TextBlock {
        let chars: Vec<TextChar> = text
            .chars()
            .enumerate()
            .map(|(i, c)| TextChar {
                char: c,
                bbox: Rect::new(x + i as f32 * 5.0, y, 5.0, 10.0),
                font_name: "Times".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
                color: Color::black(),
                mcid: None,
            })
            .collect();

        TextBlock::from_chars(chars)
    }

    #[test]
    fn test_table_format_config_default() {
        let config = TableFormatConfig::default();
        assert!(config.include_header_separator);
        assert_eq!(config.cell_padding, 1);
        assert_eq!(config.min_column_width, 3);
        assert!(config.merge_adjacent_empty_cells);
        assert!(config.preserve_cell_formatting);
        assert_eq!(config.empty_cell_text, "-");
    }

    #[test]
    fn test_table_format_config_compact() {
        let config = TableFormatConfig::compact();
        assert!(config.include_header_separator);
        assert_eq!(config.cell_padding, 0);
        assert_eq!(config.min_column_width, 1);
        assert!(config.merge_adjacent_empty_cells);
        assert!(!config.preserve_cell_formatting);
        assert_eq!(config.empty_cell_text, "");
    }

    #[test]
    fn test_table_format_config_detailed() {
        let config = TableFormatConfig::detailed();
        assert!(config.include_header_separator);
        assert_eq!(config.cell_padding, 2);
        assert_eq!(config.min_column_width, 5);
        assert!(!config.merge_adjacent_empty_cells);
        assert!(config.preserve_cell_formatting);
        assert_eq!(config.empty_cell_text, "—");
    }

    #[test]
    fn test_table_format_config_custom() {
        let config = TableFormatConfig::custom()
            .with_cell_padding(2)
            .with_min_column_width(4)
            .with_empty_cell_text("N/A");

        assert_eq!(config.cell_padding, 2);
        assert_eq!(config.min_column_width, 4);
        assert_eq!(config.empty_cell_text, "N/A");
    }

    #[test]
    fn test_markdown_table_output_format() {
        let table = Table {
            bbox: Rect::new(0.0, 0.0, 100.0, 50.0),
            cells: vec![vec![0, 1], vec![2, 3]],
            num_rows: 2,
            num_cols: 2,
        };

        let blocks = vec![
            mock_block("Header1", 0.0, 0.0),
            mock_block("Header2", 50.0, 0.0),
            mock_block("Data1", 0.0, 25.0),
            mock_block("Data2", 50.0, 25.0),
        ];

        let config = TableFormatConfig::default();
        let markdown = MarkdownTableFormatter::format_table(&table, &blocks, &config);

        assert!(markdown.contains("|"), "Markdown should contain pipe delimiters");
        assert!(markdown.contains("Header1"), "Should contain header content");
        assert!(markdown.contains("Data1"), "Should contain data content");

        let lines: Vec<&str> = markdown.lines().collect();
        assert!(lines.len() >= 3, "Should have at least header, separator, and data rows");

        if lines.len() > 1 {
            assert!(lines[1].contains("-"), "Second line should be separator with dashes");
        }
    }

    #[test]
    fn test_markdown_table_empty_cells() {
        let table = Table {
            bbox: Rect::new(0.0, 0.0, 100.0, 50.0),
            cells: vec![vec![0, 1], vec![2, 3]],
            num_rows: 2,
            num_cols: 2,
        };

        let blocks = vec![
            mock_block("Header1", 0.0, 0.0),
            mock_block("Header2", 50.0, 0.0),
            mock_block("Data1", 0.0, 25.0),
            mock_block("Data2", 50.0, 25.0),
        ];

        let config = TableFormatConfig::default();
        let markdown = MarkdownTableFormatter::format_table(&table, &blocks, &config);

        assert!(markdown.contains("Header1"), "Should contain non-empty cells");

        let config_custom = TableFormatConfig::custom().with_empty_cell_text("N/A");
        let markdown_custom = MarkdownTableFormatter::format_table(&table, &blocks, &config_custom);
        assert!(markdown_custom.contains("Header1"), "Should use custom config");
    }

    #[test]
    fn test_markdown_table_column_alignment() {
        let table = Table {
            bbox: Rect::new(0.0, 0.0, 100.0, 50.0),
            cells: vec![vec![0, 1], vec![2, 3]],
            num_rows: 2,
            num_cols: 2,
        };

        let blocks = vec![
            mock_block("Short", 0.0, 0.0),
            mock_block("VeryLongHeader", 50.0, 0.0),
            mock_block("X", 0.0, 25.0),
            mock_block("Y", 50.0, 25.0),
        ];

        let config = TableFormatConfig::default();
        let markdown = MarkdownTableFormatter::format_table(&table, &blocks, &config);

        let lines: Vec<&str> = markdown.lines().collect();
        let first_pipe_count = lines[0].matches('|').count();
        for line in &lines {
            let pipe_count = line.matches('|').count();
            assert_eq!(pipe_count, first_pipe_count);
        }
    }

    #[test]
    fn test_markdown_table_with_formatting() {
        let table = Table {
            bbox: Rect::new(0.0, 0.0, 100.0, 50.0),
            cells: vec![vec![0, 1], vec![2, 3]],
            num_rows: 2,
            num_cols: 2,
        };

        let blocks = vec![
            mock_block("**Bold**", 0.0, 0.0),
            mock_block("*Italic*", 50.0, 0.0),
            mock_block("Normal", 0.0, 25.0),
            mock_block("Data", 50.0, 25.0),
        ];

        let config_preserve = TableFormatConfig::default();
        let markdown_preserve =
            MarkdownTableFormatter::format_table(&table, &blocks, &config_preserve);
        assert!(markdown_preserve.contains("**Bold**"));

        let config_strip = TableFormatConfig::compact();
        let markdown_strip = MarkdownTableFormatter::format_table(&table, &blocks, &config_strip);
        assert!(markdown_strip.contains("Bold"));
    }

    #[test]
    fn test_markdown_table_empty_table() {
        let table = Table {
            bbox: Rect::new(0.0, 0.0, 100.0, 50.0),
            cells: vec![],
            num_rows: 0,
            num_cols: 0,
        };

        let blocks = vec![];
        let config = TableFormatConfig::default();
        let markdown = MarkdownTableFormatter::format_table(&table, &blocks, &config);

        assert_eq!(markdown, "");
    }

    #[test]
    fn test_markdown_table_single_row() {
        let table = Table {
            bbox: Rect::new(0.0, 0.0, 100.0, 20.0),
            cells: vec![vec![0, 1]],
            num_rows: 1,
            num_cols: 2,
        };

        let blocks = vec![mock_block("Col1", 0.0, 0.0), mock_block("Col2", 50.0, 0.0)];

        let config = TableFormatConfig::default();
        let markdown = MarkdownTableFormatter::format_table(&table, &blocks, &config);

        let lines: Vec<&str> = markdown.lines().collect();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_markdown_table_column_width_minimum() {
        let table = Table {
            bbox: Rect::new(0.0, 0.0, 100.0, 50.0),
            cells: vec![vec![0, 1], vec![2, 3]],
            num_rows: 2,
            num_cols: 2,
        };

        let blocks = vec![
            mock_block("A", 0.0, 0.0),
            mock_block("B", 50.0, 0.0),
            mock_block("C", 0.0, 25.0),
            mock_block("D", 50.0, 25.0),
        ];

        let config = TableFormatConfig::custom().with_min_column_width(5);

        let markdown = MarkdownTableFormatter::format_table(&table, &blocks, &config);

        let lines: Vec<&str> = markdown.lines().collect();
        if lines.len() > 1 {
            let separator_line = lines[1];
            let separator_cells: Vec<&str> = separator_line.split('|').collect();

            for cell in separator_cells {
                let dashes = cell.matches('-').count();
                if dashes > 0 {
                    assert!(dashes >= 3);
                }
            }
        }
    }
}
