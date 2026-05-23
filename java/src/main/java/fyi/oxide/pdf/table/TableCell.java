/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.table;

import fyi.oxide.pdf.geometry.BBox;
import java.util.Objects;

/**
 * A single cell in an extracted {@link Table}. Cells may span
 * multiple rows ({@link #rowSpan()}) or columns ({@link #colSpan()}).
 */
public final class TableCell {
    private final String text;
    private final BBox bbox;
    private final int row;
    private final int col;
    private final int rowSpan;
    private final int colSpan;

    public TableCell(String text, BBox bbox, int row, int col, int rowSpan, int colSpan) {
        this.text = Objects.requireNonNull(text, "text");
        this.bbox = Objects.requireNonNull(bbox, "bbox");
        this.row = row;
        this.col = col;
        this.rowSpan = rowSpan;
        this.colSpan = colSpan;
    }

    public String text() {
        return text;
    }

    public BBox bbox() {
        return bbox;
    }
    /** @return 0-based row index of the cell's top-left anchor. */
    public int row() {
        return row;
    }
    /** @return 0-based column index of the cell's top-left anchor. */
    public int col() {
        return col;
    }
    /** @return number of rows this cell spans (&ge;1). */
    public int rowSpan() {
        return rowSpan;
    }
    /** @return number of columns this cell spans (&ge;1). */
    public int colSpan() {
        return colSpan;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TableCell)) return false;
        TableCell c = (TableCell) o;
        return row == c.row
                && col == c.col
                && rowSpan == c.rowSpan
                && colSpan == c.colSpan
                && text.equals(c.text)
                && bbox.equals(c.bbox);
    }

    @Override
    public int hashCode() {
        return Objects.hash(text, bbox, row, col, rowSpan, colSpan);
    }

    @Override
    public String toString() {
        return "TableCell[(" + row + "," + col + ")"
                + (rowSpan == 1 && colSpan == 1 ? "" : " span=(" + rowSpan + "," + colSpan + ")")
                + " text=" + text + "]";
    }
}
