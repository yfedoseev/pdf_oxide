/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.table;

import fyi.oxide.pdf.geometry.BBox;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A table extracted from a PDF page. Composed of {@link TableCell}s
 * with row/column indices that may have non-trivial row/col spans.
 *
 * <p>v0.3.53 ships the native grid-detector output (the same the
 * other 7 bindings expose). For image-tables reconstructed via OCR
 * + spatial detector (the v0.3.51 AutoExtractor path), use
 * {@link fyi.oxide.pdf.auto.RegionResult#table()}.
 */
public final class Table {
    private final BBox bbox;
    private final int rows;
    private final int cols;
    private final List<TableCell> cells;

    public Table(BBox bbox, int rows, int cols, List<TableCell> cells) {
        this.bbox = Objects.requireNonNull(bbox, "bbox");
        this.rows = rows;
        this.cols = cols;
        this.cells = Collections.unmodifiableList(new java.util.ArrayList<>(Objects.requireNonNull(cells, "cells")));
    }

    public BBox bbox() {
        return bbox;
    }
    /** @return number of rows (max row index + 1). */
    public int rows() {
        return rows;
    }
    /** @return number of columns (max col index + 1). */
    public int cols() {
        return cols;
    }
    /** @return unmodifiable view of all cells in row-major order. */
    public List<TableCell> cells() {
        return cells;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Table)) return false;
        Table t = (Table) o;
        return rows == t.rows && cols == t.cols && bbox.equals(t.bbox) && cells.equals(t.cells);
    }

    @Override
    public int hashCode() {
        return Objects.hash(bbox, rows, cols, cells);
    }

    @Override
    public String toString() {
        return "Table[" + rows + "x" + cols + " " + cells.size() + " cells, bbox=" + bbox + "]";
    }
}
