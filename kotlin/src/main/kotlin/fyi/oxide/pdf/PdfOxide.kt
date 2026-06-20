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
