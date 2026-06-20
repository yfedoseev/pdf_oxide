// pdf_oxide — idiomatic Kotlin bindings over the C ABI via JNA.
//
// Pure-Kotlin FFI: JNA binds the cdylib (libpdf_oxide) by name. Handles are
// JNA Pointers wrapped in AutoCloseable classes; returned C strings/buffers are
// copied into Kotlin and freed via free_string; non-success C-ABI error codes
// throw PdfOxideException.
//
// API surface mirrors the other language bindings; coverage is asserted by
// ApiCoverageTest (one test per public method).
package fyi.oxide.pdf

import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Pointer
import com.sun.jna.ptr.ByteByReference
import com.sun.jna.ptr.FloatByReference
import com.sun.jna.ptr.IntByReference

/** Thrown on any non-success C-ABI error code. */
class PdfOxideException(
    val code: Int,
    val op: String,
) : RuntimeException("pdf_oxide: $op failed (error code $code)")

/** PDF version (e.g. 1.7). */
data class PdfVersion(
    val major: Int,
    val minor: Int,
) {
    override fun toString() = "$major.$minor"
}

/** An axis-aligned bounding box in PDF user-space units. */
data class Bbox(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
)

/** A single extracted glyph. [character] is the Unicode codepoint. */
data class Char(
    val character: Int,
    val bbox: Bbox,
    val fontName: String,
    val fontSize: Float,
)

/** A single extracted word. */
data class Word(
    val text: String,
    val bbox: Bbox,
    val fontName: String,
    val fontSize: Float,
    val bold: Boolean,
)

/** A single extracted line of text. */
data class TextLine(
    val text: String,
    val bbox: Bbox,
    val wordCount: Int,
)

/** A single extracted table; cells are addressed by [cell]. */
data class Table(
    val rowCount: Int,
    val colCount: Int,
    val hasHeader: Boolean,
    private val cells: List<List<String>>,
) {
    /** The text of the cell at 0-based [row], [col]. */
    fun cell(
        row: Int,
        col: Int,
    ): String = cells[row][col]
}

/** An embedded font on a page. */
data class Font(
    val name: String,
    val type: String,
    val encoding: String,
    val embedded: Boolean,
    val subset: Boolean,
)

/** An embedded image on a page; [data] is the raw image bytes. */
data class Image(
    val width: Int,
    val height: Int,
    val bitsPerComponent: Int,
    val format: String,
    val colorspace: String,
    val data: ByteArray,
)

/**
 * A rendered page image. Owns the native FfiRenderedImage handle: [width],
 * [height] and [data] (the encoded image bytes) are read eagerly at
 * construction, while [save] uses the live handle. Close when done
 * (AutoCloseable) to free the native handle; a finalizer is a backstop.
 */
class RenderedImage internal constructor(
    private var handle: Pointer?,
    val width: Int,
    val height: Int,
    val data: ByteArray,
) : AutoCloseable {
    private fun ptr(): Pointer = handle ?: error("RenderedImage is closed")

    /** Write the rendered image to [path], encoded per the render format. */
    fun save(path: String) {
        val code = IntByReference()
        if (Native_.lib.pdf_save_rendered_image(ptr(), path, code) != 0) {
            throw PdfOxideException(code.value, "saveRenderedImage")
        }
    }

    override fun close() {
        handle?.let {
            Native_.lib.pdf_rendered_image_free(it)
            handle = null
        }
    }

    @Suppress("ProtectedInFinal")
    protected fun finalize() {
        close()
    }
}

/** A page annotation. */
data class Annotation(
    val type: String,
    val subtype: String,
    val content: String,
    val author: String,
    val rect: Bbox,
    val borderWidth: Float,
)

/** A vector path on a page. */
data class Path(
    val bbox: Bbox,
    val strokeWidth: Float,
    val hasStroke: Boolean,
    val hasFill: Boolean,
    val operationCount: Int,
)

/** A single text search hit. */
data class SearchResult(
    val text: String,
    val page: Int,
    val bbox: Bbox,
)

/** Raw JNA binding to the pdf_oxide C ABI (internal). */
internal interface CLib : Library {
    fun pdf_document_open(
        path: String,
        code: IntByReference,
    ): Pointer?

    fun pdf_document_open_from_bytes(
        data: ByteArray,
        len: Long,
        code: IntByReference,
    ): Pointer?

    fun pdf_document_open_with_password(
        path: String,
        pw: String,
        code: IntByReference,
    ): Pointer?

    fun pdf_document_free(h: Pointer)

    fun pdf_document_get_page_count(
        h: Pointer,
        code: IntByReference,
    ): Int

    fun pdf_document_get_version(
        h: Pointer,
        major: ByteByReference,
        minor: ByteByReference,
    )

    fun pdf_document_is_encrypted(h: Pointer): Boolean

    fun pdf_document_has_structure_tree(h: Pointer): Boolean

    fun pdf_document_extract_text(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_document_to_plain_text(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_document_to_markdown(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_document_to_html(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_document_to_markdown_all(
        h: Pointer,
        code: IntByReference,
    ): Pointer?

    fun pdf_document_to_html_all(
        h: Pointer,
        code: IntByReference,
    ): Pointer?

    fun pdf_document_to_plain_text_all(
        h: Pointer,
        code: IntByReference,
    ): Pointer?

    fun pdf_document_authenticate(
        h: Pointer,
        password: String,
        code: IntByReference,
    ): Boolean

    fun pdf_document_extract_structured_to_json(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    // ── Phase-1 element extraction: chars ─────────────────────────────────────
    fun pdf_document_extract_chars(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_char_count(list: Pointer): Int

    fun pdf_oxide_char_get_char(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Int

    fun pdf_oxide_char_get_bbox(
        list: Pointer,
        index: Int,
        x: FloatByReference,
        y: FloatByReference,
        w: FloatByReference,
        h: FloatByReference,
        code: IntByReference,
    )

    fun pdf_oxide_char_get_font_name(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_char_get_font_size(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Float

    fun pdf_oxide_char_list_free(list: Pointer)

    // ── Phase-1 element extraction: words ─────────────────────────────────────
    fun pdf_document_extract_words(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_word_count(list: Pointer): Int

    fun pdf_oxide_word_get_text(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_word_get_bbox(
        list: Pointer,
        index: Int,
        x: FloatByReference,
        y: FloatByReference,
        w: FloatByReference,
        h: FloatByReference,
        code: IntByReference,
    )

    fun pdf_oxide_word_get_font_name(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_word_get_font_size(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Float

    fun pdf_oxide_word_is_bold(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Boolean

    fun pdf_oxide_word_list_free(list: Pointer)

    // ── Phase-1 element extraction: text lines ────────────────────────────────
    fun pdf_document_extract_text_lines(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_line_count(list: Pointer): Int

    fun pdf_oxide_line_get_text(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_line_get_bbox(
        list: Pointer,
        index: Int,
        x: FloatByReference,
        y: FloatByReference,
        w: FloatByReference,
        h: FloatByReference,
        code: IntByReference,
    )

    fun pdf_oxide_line_get_word_count(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Int

    fun pdf_oxide_line_list_free(list: Pointer)

    // ── Phase-1 element extraction: tables ────────────────────────────────────
    fun pdf_document_extract_tables(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_table_count(list: Pointer): Int

    fun pdf_oxide_table_get_row_count(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Int

    fun pdf_oxide_table_get_col_count(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Int

    fun pdf_oxide_table_get_cell_text(
        list: Pointer,
        tableIndex: Int,
        row: Int,
        col: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_table_has_header(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Boolean

    fun pdf_oxide_table_list_free(list: Pointer)

    // ── Phase-2 element extraction: fonts ─────────────────────────────────────
    fun pdf_document_get_embedded_fonts(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_font_count(list: Pointer): Int

    fun pdf_oxide_font_get_name(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_font_get_type(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_font_get_encoding(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_font_is_embedded(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Boolean

    fun pdf_oxide_font_is_subset(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Boolean

    fun pdf_oxide_font_list_free(list: Pointer)

    // ── Phase-2 element extraction: images ────────────────────────────────────
    fun pdf_document_get_embedded_images(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_image_count(list: Pointer): Int

    fun pdf_oxide_image_get_width(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Int

    fun pdf_oxide_image_get_height(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Int

    fun pdf_oxide_image_get_bits_per_component(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Int

    fun pdf_oxide_image_get_format(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_image_get_colorspace(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_image_get_data(
        list: Pointer,
        index: Int,
        dataLen: IntByReference,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_image_list_free(list: Pointer)

    // ── Phase-2 element extraction: annotations ───────────────────────────────
    fun pdf_document_get_page_annotations(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_annotation_count(list: Pointer): Int

    fun pdf_oxide_annotation_get_type(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_annotation_get_subtype(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_annotation_get_content(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_annotation_get_author(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_annotation_get_rect(
        list: Pointer,
        index: Int,
        x: FloatByReference,
        y: FloatByReference,
        w: FloatByReference,
        h: FloatByReference,
        code: IntByReference,
    )

    fun pdf_oxide_annotation_get_border_width(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Float

    fun pdf_oxide_annotation_list_free(list: Pointer)

    // ── Phase-2 element extraction: paths ─────────────────────────────────────
    fun pdf_document_extract_paths(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_path_count(list: Pointer): Int

    fun pdf_oxide_path_get_bbox(
        list: Pointer,
        index: Int,
        x: FloatByReference,
        y: FloatByReference,
        w: FloatByReference,
        h: FloatByReference,
        code: IntByReference,
    )

    fun pdf_oxide_path_get_stroke_width(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Float

    fun pdf_oxide_path_has_stroke(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Boolean

    fun pdf_oxide_path_has_fill(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Boolean

    fun pdf_oxide_path_get_operation_count(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Int

    fun pdf_oxide_path_list_free(list: Pointer)

    // ── Phase-2 element extraction: search ────────────────────────────────────
    fun pdf_document_search_page(
        h: Pointer,
        page: Int,
        term: String,
        caseSensitive: Boolean,
        code: IntByReference,
    ): Pointer?

    fun pdf_document_search_all(
        h: Pointer,
        term: String,
        caseSensitive: Boolean,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_search_result_count(list: Pointer): Int

    fun pdf_oxide_search_result_get_text(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_oxide_search_result_get_page(
        list: Pointer,
        index: Int,
        code: IntByReference,
    ): Int

    fun pdf_oxide_search_result_get_bbox(
        list: Pointer,
        index: Int,
        x: FloatByReference,
        y: FloatByReference,
        w: FloatByReference,
        h: FloatByReference,
        code: IntByReference,
    )

    fun pdf_oxide_search_result_free(list: Pointer)

    fun pdf_from_markdown(
        md: String,
        code: IntByReference,
    ): Pointer?

    fun pdf_from_html(
        html: String,
        code: IntByReference,
    ): Pointer?

    fun pdf_from_text(
        text: String,
        code: IntByReference,
    ): Pointer?

    fun pdf_free(h: Pointer)

    fun pdf_save(
        h: Pointer,
        path: String,
        code: IntByReference,
    ): Int

    fun pdf_save_to_bytes(
        h: Pointer,
        len: IntByReference,
        code: IntByReference,
    ): Pointer?

    fun free_string(p: Pointer)

    fun free_bytes(p: Pointer)

    // ── Phase-3 page rendering ────────────────────────────────────────────────
    fun pdf_render_page(
        doc: Pointer,
        pageIndex: Int,
        format: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_render_page_zoom(
        doc: Pointer,
        pageIndex: Int,
        zoom: Float,
        format: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_render_page_thumbnail(
        doc: Pointer,
        pageIndex: Int,
        size: Int,
        format: Int,
        code: IntByReference,
    ): Pointer?

    fun pdf_get_rendered_image_width(
        img: Pointer,
        code: IntByReference,
    ): Int

    fun pdf_get_rendered_image_height(
        img: Pointer,
        code: IntByReference,
    ): Int

    fun pdf_get_rendered_image_data(
        img: Pointer,
        dataLen: IntByReference,
        code: IntByReference,
    ): Pointer?

    fun pdf_save_rendered_image(
        img: Pointer,
        filePath: String,
        code: IntByReference,
    ): Int

    fun pdf_rendered_image_free(handle: Pointer)
}

internal object Native_ {
    val lib: CLib = Native.load("pdf_oxide", CLib::class.java)

    fun takeString(
        p: Pointer?,
        code: Int,
        op: String,
    ): String {
        if (p == null) throw PdfOxideException(code, op)
        val s = p.getString(0)
        lib.free_string(p)
        return s
    }

    /** Read a string out-param via [code], throwing on error. */
    fun readString(
        p: Pointer?,
        code: IntByReference,
        op: String,
    ): String {
        if (code.value != 0) throw PdfOxideException(code.value, op)
        return takeString(p, code.value, op)
    }
}

/** An opened PDF for extraction/inspection. Close when done (AutoCloseable). */
class PdfDocument internal constructor(
    private var handle: Pointer?,
) : AutoCloseable {
    private fun ptr(): Pointer = handle ?: error("PdfDocument is closed")

    companion object {
        @JvmStatic
        fun open(path: String): PdfDocument {
            val code = IntByReference()
            val h =
                Native_.lib.pdf_document_open(path, code)
                    ?: throw PdfOxideException(code.value, "open")
            return PdfDocument(h)
        }

        @JvmStatic
        fun openFromBytes(data: ByteArray): PdfDocument {
            val code = IntByReference()
            val h =
                Native_.lib.pdf_document_open_from_bytes(data, data.size.toLong(), code)
                    ?: throw PdfOxideException(code.value, "openFromBytes")
            return PdfDocument(h)
        }

        @JvmStatic
        fun openWithPassword(
            path: String,
            password: String,
        ): PdfDocument {
            val code = IntByReference()
            val h =
                Native_.lib.pdf_document_open_with_password(path, password, code)
                    ?: throw PdfOxideException(code.value, "openWithPassword")
            return PdfDocument(h)
        }
    }

    fun pageCount(): Int {
        val code = IntByReference()
        val n = Native_.lib.pdf_document_get_page_count(ptr(), code)
        if (n < 0) throw PdfOxideException(code.value, "pageCount")
        return n
    }

    fun version(): PdfVersion {
        val maj = ByteByReference()
        val min = ByteByReference()
        Native_.lib.pdf_document_get_version(ptr(), maj, min)
        return PdfVersion(maj.value.toInt() and 0xFF, min.value.toInt() and 0xFF)
    }

    fun isEncrypted(): Boolean = Native_.lib.pdf_document_is_encrypted(ptr())

    fun hasStructureTree(): Boolean = Native_.lib.pdf_document_has_structure_tree(ptr())

    fun extractText(page: Int): String {
        val code = IntByReference()
        return Native_.takeString(Native_.lib.pdf_document_extract_text(ptr(), page, code), code.value, "extractText")
    }

    fun toPlainText(page: Int): String {
        val code = IntByReference()
        return Native_.takeString(Native_.lib.pdf_document_to_plain_text(ptr(), page, code), code.value, "toPlainText")
    }

    fun toMarkdown(page: Int): String {
        val code = IntByReference()
        return Native_.takeString(Native_.lib.pdf_document_to_markdown(ptr(), page, code), code.value, "toMarkdown")
    }

    fun toHtml(page: Int): String {
        val code = IntByReference()
        return Native_.takeString(Native_.lib.pdf_document_to_html(ptr(), page, code), code.value, "toHtml")
    }

    fun toMarkdownAll(): String {
        val code = IntByReference()
        return Native_.takeString(Native_.lib.pdf_document_to_markdown_all(ptr(), code), code.value, "toMarkdownAll")
    }

    fun toHtmlAll(): String {
        val code = IntByReference()
        return Native_.takeString(Native_.lib.pdf_document_to_html_all(ptr(), code), code.value, "toHtmlAll")
    }

    fun toPlainTextAll(): String {
        val code = IntByReference()
        return Native_.takeString(Native_.lib.pdf_document_to_plain_text_all(ptr(), code), code.value, "toPlainTextAll")
    }

    /**
     * Verify [password] against an encrypted document. Returns true on success,
     * false for a wrong password (this is not an error — mirrors the other bool
     * accessors, which never throw).
     */
    fun authenticate(password: String): Boolean {
        val code = IntByReference()
        return Native_.lib.pdf_document_authenticate(ptr(), password, code)
    }

    /** A handle to a single 0-based page, kept alive by this document. */
    fun page(index: Int): Page = Page(this, index)

    fun extractStructuredJson(page: Int): String {
        val code = IntByReference()
        return Native_.takeString(
            Native_.lib.pdf_document_extract_structured_to_json(ptr(), page, code),
            code.value,
            "extractStructuredJson",
        )
    }

    /** Extract individual glyphs from the 0-based [pageIndex]. */
    @Suppress("ThrowsCount")
    fun extractChars(pageIndex: Int): List<Char> {
        val code = IntByReference()
        val list =
            Native_.lib.pdf_document_extract_chars(ptr(), pageIndex, code)
                ?: throw PdfOxideException(code.value, "extractChars")
        try {
            val n = Native_.lib.pdf_oxide_char_count(list)
            val out = ArrayList<Char>(if (n < 0) 0 else n)
            for (i in 0 until n) {
                val ch = Native_.lib.pdf_oxide_char_get_char(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractChars")
                val bbox = readBbox(code, "extractChars") { x, y, w, h, c -> Native_.lib.pdf_oxide_char_get_bbox(list, i, x, y, w, h, c) }
                val fontName = Native_.readString(Native_.lib.pdf_oxide_char_get_font_name(list, i, code), code, "extractChars")
                val fontSize = Native_.lib.pdf_oxide_char_get_font_size(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractChars")
                out.add(Char(ch, bbox, fontName, fontSize))
            }
            return out
        } finally {
            Native_.lib.pdf_oxide_char_list_free(list)
        }
    }

    /** Extract words from the 0-based [pageIndex]. */
    @Suppress("ThrowsCount")
    fun extractWords(pageIndex: Int): List<Word> {
        val code = IntByReference()
        val list =
            Native_.lib.pdf_document_extract_words(ptr(), pageIndex, code)
                ?: throw PdfOxideException(code.value, "extractWords")
        try {
            val n = Native_.lib.pdf_oxide_word_count(list)
            val out = ArrayList<Word>(if (n < 0) 0 else n)
            for (i in 0 until n) {
                val text = Native_.readString(Native_.lib.pdf_oxide_word_get_text(list, i, code), code, "extractWords")
                val bbox = readBbox(code, "extractWords") { x, y, w, h, c -> Native_.lib.pdf_oxide_word_get_bbox(list, i, x, y, w, h, c) }
                val fontName = Native_.readString(Native_.lib.pdf_oxide_word_get_font_name(list, i, code), code, "extractWords")
                val fontSize = Native_.lib.pdf_oxide_word_get_font_size(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractWords")
                val bold = Native_.lib.pdf_oxide_word_is_bold(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractWords")
                out.add(Word(text, bbox, fontName, fontSize, bold))
            }
            return out
        } finally {
            Native_.lib.pdf_oxide_word_list_free(list)
        }
    }

    /** Extract text lines from the 0-based [pageIndex]. */
    @Suppress("ThrowsCount")
    fun extractTextLines(pageIndex: Int): List<TextLine> {
        val code = IntByReference()
        val list =
            Native_.lib.pdf_document_extract_text_lines(ptr(), pageIndex, code)
                ?: throw PdfOxideException(code.value, "extractTextLines")
        try {
            val n = Native_.lib.pdf_oxide_line_count(list)
            val out = ArrayList<TextLine>(if (n < 0) 0 else n)
            for (i in 0 until n) {
                val text = Native_.readString(Native_.lib.pdf_oxide_line_get_text(list, i, code), code, "extractTextLines")
                val bbox =
                    readBbox(code, "extractTextLines") { x, y, w, h, c -> Native_.lib.pdf_oxide_line_get_bbox(list, i, x, y, w, h, c) }
                val wordCount = Native_.lib.pdf_oxide_line_get_word_count(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractTextLines")
                out.add(TextLine(text, bbox, wordCount))
            }
            return out
        } finally {
            Native_.lib.pdf_oxide_line_list_free(list)
        }
    }

    /** Extract tables from the 0-based [pageIndex]. */
    @Suppress("ThrowsCount")
    fun extractTables(pageIndex: Int): List<Table> {
        val code = IntByReference()
        val list =
            Native_.lib.pdf_document_extract_tables(ptr(), pageIndex, code)
                ?: throw PdfOxideException(code.value, "extractTables")
        try {
            val n = Native_.lib.pdf_oxide_table_count(list)
            val out = ArrayList<Table>(if (n < 0) 0 else n)
            for (i in 0 until n) {
                val rows = Native_.lib.pdf_oxide_table_get_row_count(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractTables")
                val cols = Native_.lib.pdf_oxide_table_get_col_count(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractTables")
                val hasHeader = Native_.lib.pdf_oxide_table_has_header(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractTables")
                val cells =
                    (0 until rows).map { r ->
                        (0 until cols).map { c ->
                            Native_.readString(Native_.lib.pdf_oxide_table_get_cell_text(list, i, r, c, code), code, "extractTables")
                        }
                    }
                out.add(Table(rows, cols, hasHeader, cells))
            }
            return out
        } finally {
            Native_.lib.pdf_oxide_table_list_free(list)
        }
    }

    /** Extract embedded fonts from the 0-based [pageIndex]. */
    @Suppress("ThrowsCount")
    fun embeddedFonts(pageIndex: Int): List<Font> {
        val code = IntByReference()
        val list =
            Native_.lib.pdf_document_get_embedded_fonts(ptr(), pageIndex, code)
                ?: throw PdfOxideException(code.value, "embeddedFonts")
        try {
            val n = Native_.lib.pdf_oxide_font_count(list)
            val out = ArrayList<Font>(if (n < 0) 0 else n)
            for (i in 0 until n) {
                val name = Native_.readString(Native_.lib.pdf_oxide_font_get_name(list, i, code), code, "embeddedFonts")
                val type = Native_.readString(Native_.lib.pdf_oxide_font_get_type(list, i, code), code, "embeddedFonts")
                val encoding = Native_.readString(Native_.lib.pdf_oxide_font_get_encoding(list, i, code), code, "embeddedFonts")
                val embedded = Native_.lib.pdf_oxide_font_is_embedded(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "embeddedFonts")
                val subset = Native_.lib.pdf_oxide_font_is_subset(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "embeddedFonts")
                out.add(Font(name, type, encoding, embedded, subset))
            }
            return out
        } finally {
            Native_.lib.pdf_oxide_font_list_free(list)
        }
    }

    /** Extract embedded images from the 0-based [pageIndex]. */
    @Suppress("ThrowsCount")
    fun embeddedImages(pageIndex: Int): List<Image> {
        val code = IntByReference()
        val list =
            Native_.lib.pdf_document_get_embedded_images(ptr(), pageIndex, code)
                ?: throw PdfOxideException(code.value, "embeddedImages")
        try {
            val n = Native_.lib.pdf_oxide_image_count(list)
            val out = ArrayList<Image>(if (n < 0) 0 else n)
            for (i in 0 until n) {
                val width = Native_.lib.pdf_oxide_image_get_width(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "embeddedImages")
                val height = Native_.lib.pdf_oxide_image_get_height(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "embeddedImages")
                val bpc = Native_.lib.pdf_oxide_image_get_bits_per_component(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "embeddedImages")
                val format = Native_.readString(Native_.lib.pdf_oxide_image_get_format(list, i, code), code, "embeddedImages")
                val colorspace = Native_.readString(Native_.lib.pdf_oxide_image_get_colorspace(list, i, code), code, "embeddedImages")
                val len = IntByReference()
                val p =
                    Native_.lib.pdf_oxide_image_get_data(list, i, len, code)
                        ?: throw PdfOxideException(code.value, "embeddedImages")
                val nBytes = if (len.value < 0) 0 else len.value
                val data = p.getByteArray(0, nBytes)
                Native_.lib.free_bytes(p)
                out.add(Image(width, height, bpc, format, colorspace, data))
            }
            return out
        } finally {
            Native_.lib.pdf_oxide_image_list_free(list)
        }
    }

    /** Extract annotations from the 0-based [pageIndex]. */
    @Suppress("ThrowsCount")
    fun pageAnnotations(pageIndex: Int): List<Annotation> {
        val code = IntByReference()
        val list =
            Native_.lib.pdf_document_get_page_annotations(ptr(), pageIndex, code)
                ?: throw PdfOxideException(code.value, "pageAnnotations")
        try {
            val n = Native_.lib.pdf_oxide_annotation_count(list)
            val out = ArrayList<Annotation>(if (n < 0) 0 else n)
            for (i in 0 until n) {
                val type = Native_.readString(Native_.lib.pdf_oxide_annotation_get_type(list, i, code), code, "pageAnnotations")
                val subtype = Native_.readString(Native_.lib.pdf_oxide_annotation_get_subtype(list, i, code), code, "pageAnnotations")
                val content = Native_.readString(Native_.lib.pdf_oxide_annotation_get_content(list, i, code), code, "pageAnnotations")
                val author = Native_.readString(Native_.lib.pdf_oxide_annotation_get_author(list, i, code), code, "pageAnnotations")
                val rect =
                    readBbox(code, "pageAnnotations") { x, y, w, h, c -> Native_.lib.pdf_oxide_annotation_get_rect(list, i, x, y, w, h, c) }
                val borderWidth = Native_.lib.pdf_oxide_annotation_get_border_width(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "pageAnnotations")
                out.add(Annotation(type, subtype, content, author, rect, borderWidth))
            }
            return out
        } finally {
            Native_.lib.pdf_oxide_annotation_list_free(list)
        }
    }

    /** Extract vector paths from the 0-based [pageIndex]. */
    @Suppress("ThrowsCount")
    fun extractPaths(pageIndex: Int): List<Path> {
        val code = IntByReference()
        val list =
            Native_.lib.pdf_document_extract_paths(ptr(), pageIndex, code)
                ?: throw PdfOxideException(code.value, "extractPaths")
        try {
            val n = Native_.lib.pdf_oxide_path_count(list)
            val out = ArrayList<Path>(if (n < 0) 0 else n)
            for (i in 0 until n) {
                val bbox = readBbox(code, "extractPaths") { x, y, w, h, c -> Native_.lib.pdf_oxide_path_get_bbox(list, i, x, y, w, h, c) }
                val strokeWidth = Native_.lib.pdf_oxide_path_get_stroke_width(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractPaths")
                val hasStroke = Native_.lib.pdf_oxide_path_has_stroke(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractPaths")
                val hasFill = Native_.lib.pdf_oxide_path_has_fill(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractPaths")
                val operationCount = Native_.lib.pdf_oxide_path_get_operation_count(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, "extractPaths")
                out.add(Path(bbox, strokeWidth, hasStroke, hasFill, operationCount))
            }
            return out
        } finally {
            Native_.lib.pdf_oxide_path_list_free(list)
        }
    }

    /** Search the 0-based [pageIndex] for [term]. */
    @Suppress("ThrowsCount")
    fun search(
        pageIndex: Int,
        term: String,
        caseSensitive: Boolean,
    ): List<SearchResult> {
        val code = IntByReference()
        val list =
            Native_.lib.pdf_document_search_page(ptr(), pageIndex, term, caseSensitive, code)
                ?: throw PdfOxideException(code.value, "search")
        return readSearchResults(list, code, "search")
    }

    /** Search all pages for [term]. */
    @Suppress("ThrowsCount")
    fun searchAll(
        term: String,
        caseSensitive: Boolean,
    ): List<SearchResult> {
        val code = IntByReference()
        val list =
            Native_.lib.pdf_document_search_all(ptr(), term, caseSensitive, code)
                ?: throw PdfOxideException(code.value, "searchAll")
        return readSearchResults(list, code, "searchAll")
    }

    /** Render the 0-based [pageIndex] to an image ([format] 0 = PNG). */
    fun renderPage(
        pageIndex: Int,
        format: Int = 0,
    ): RenderedImage {
        val code = IntByReference()
        val img =
            Native_.lib.pdf_render_page(ptr(), pageIndex, format, code)
                ?: throw PdfOxideException(code.value, "renderPage")
        return readRenderedImage(img, "renderPage")
    }

    /** Render the 0-based [pageIndex] at [zoom] scale ([format] 0 = PNG). */
    fun renderPageZoom(
        pageIndex: Int,
        zoom: Float,
        format: Int = 0,
    ): RenderedImage {
        val code = IntByReference()
        val img =
            Native_.lib.pdf_render_page_zoom(ptr(), pageIndex, zoom, format, code)
                ?: throw PdfOxideException(code.value, "renderPageZoom")
        return readRenderedImage(img, "renderPageZoom")
    }

    /**
     * Render the 0-based [pageIndex] as a thumbnail fitting within [size] pixels
     * ([format] 0 = PNG).
     */
    fun renderPageThumbnail(
        pageIndex: Int,
        size: Int,
        format: Int = 0,
    ): RenderedImage {
        val code = IntByReference()
        val img =
            Native_.lib.pdf_render_page_thumbnail(ptr(), pageIndex, size, format, code)
                ?: throw PdfOxideException(code.value, "renderPageThumbnail")
        return readRenderedImage(img, "renderPageThumbnail")
    }

    /**
     * Read width/height/data from a live FfiRenderedImage handle and wrap it in
     * a [RenderedImage] (which takes ownership of the handle). On any error the
     * handle is freed before throwing, so no leak occurs.
     */
    @Suppress("ThrowsCount", "TooGenericExceptionCaught")
    private fun readRenderedImage(
        img: Pointer,
        op: String,
    ): RenderedImage {
        val code = IntByReference()
        try {
            val width = Native_.lib.pdf_get_rendered_image_width(img, code)
            if (code.value != 0) throw PdfOxideException(code.value, op)
            val height = Native_.lib.pdf_get_rendered_image_height(img, code)
            if (code.value != 0) throw PdfOxideException(code.value, op)
            val len = IntByReference()
            val p =
                Native_.lib.pdf_get_rendered_image_data(img, len, code)
                    ?: throw PdfOxideException(code.value, op)
            val nBytes = if (len.value < 0) 0 else len.value
            val data = p.getByteArray(0, nBytes)
            Native_.lib.free_bytes(p)
            return RenderedImage(img, width, height, data)
        } catch (e: Throwable) {
            Native_.lib.pdf_rendered_image_free(img)
            throw e
        }
    }

    @Suppress("ThrowsCount")
    private fun readSearchResults(
        list: Pointer,
        code: IntByReference,
        op: String,
    ): List<SearchResult> {
        try {
            val n = Native_.lib.pdf_oxide_search_result_count(list)
            val out = ArrayList<SearchResult>(if (n < 0) 0 else n)
            for (i in 0 until n) {
                val text = Native_.readString(Native_.lib.pdf_oxide_search_result_get_text(list, i, code), code, op)
                val page = Native_.lib.pdf_oxide_search_result_get_page(list, i, code)
                if (code.value != 0) throw PdfOxideException(code.value, op)
                val bbox = readBbox(code, op) { x, y, w, h, c -> Native_.lib.pdf_oxide_search_result_get_bbox(list, i, x, y, w, h, c) }
                out.add(SearchResult(text, page, bbox))
            }
            return out
        } finally {
            Native_.lib.pdf_oxide_search_result_free(list)
        }
    }

    private inline fun readBbox(
        code: IntByReference,
        op: String,
        getter: (FloatByReference, FloatByReference, FloatByReference, FloatByReference, IntByReference) -> Unit,
    ): Bbox {
        val x = FloatByReference()
        val y = FloatByReference()
        val w = FloatByReference()
        val h = FloatByReference()
        getter(x, y, w, h, code)
        if (code.value != 0) throw PdfOxideException(code.value, op)
        return Bbox(x.value, y.value, w.value, h.value)
    }

    override fun close() {
        handle?.let {
            Native_.lib.pdf_document_free(it)
            handle = null
        }
    }
}

/**
 * A single 0-based page of a [PdfDocument]. Holds a strong reference to its
 * document, so the document is kept alive for as long as the page is used; all
 * extraction delegates to the document's per-page methods (which enforce the
 * closed-handle guard).
 */
class Page internal constructor(
    private val document: PdfDocument,
    val index: Int,
) {
    fun text(): String = document.extractText(index)

    fun markdown(): String = document.toMarkdown(index)

    fun html(): String = document.toHtml(index)

    fun plainText(): String = document.toPlainText(index)
}

/** A PDF produced by a builder. Close when done (AutoCloseable). */
class Pdf internal constructor(
    private var handle: Pointer?,
) : AutoCloseable {
    private fun ptr(): Pointer = handle ?: error("Pdf is closed")

    companion object {
        @JvmStatic
        fun fromMarkdown(md: String): Pdf {
            val code = IntByReference()
            return Native_.lib.pdf_from_markdown(md, code)?.let { Pdf(it) }
                ?: throw PdfOxideException(code.value, "fromMarkdown")
        }

        @JvmStatic
        fun fromHtml(html: String): Pdf {
            val code = IntByReference()
            return Native_.lib.pdf_from_html(html, code)?.let { Pdf(it) }
                ?: throw PdfOxideException(code.value, "fromHtml")
        }

        @JvmStatic
        fun fromText(text: String): Pdf {
            val code = IntByReference()
            return Native_.lib.pdf_from_text(text, code)?.let { Pdf(it) }
                ?: throw PdfOxideException(code.value, "fromText")
        }
    }

    fun save(path: String) {
        val code = IntByReference()
        if (Native_.lib.pdf_save(ptr(), path, code) != 0) throw PdfOxideException(code.value, "save")
    }

    fun toBytes(): ByteArray {
        val len = IntByReference()
        val code = IntByReference()
        val p =
            Native_.lib.pdf_save_to_bytes(ptr(), len, code)
                ?: throw PdfOxideException(code.value, "toBytes")
        val n = if (len.value < 0) 0 else len.value
        val out = p.getByteArray(0, n)
        Native_.lib.free_bytes(p)
        return out
    }

    override fun close() {
        handle?.let {
            Native_.lib.pdf_free(it)
            handle = null
        }
    }
}
