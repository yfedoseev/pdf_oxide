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
import com.sun.jna.ptr.DoubleByReference
import com.sun.jna.ptr.FloatByReference
import com.sun.jna.ptr.IntByReference
import com.sun.jna.ptr.LongByReference

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

    // ── DocumentEditor: lifecycle ─────────────────────────────────────────────
    fun document_editor_open(
        path: String,
        code: IntByReference,
    ): Pointer?

    fun document_editor_open_from_bytes(
        data: ByteArray,
        len: Long,
        code: IntByReference,
    ): Pointer?

    fun document_editor_free(handle: Pointer)

    // ── DocumentEditor: inspection ────────────────────────────────────────────
    fun document_editor_is_modified(handle: Pointer): Boolean

    fun document_editor_get_source_path(
        handle: Pointer,
        code: IntByReference,
    ): Pointer?

    fun document_editor_get_version(
        handle: Pointer,
        major: ByteByReference,
        minor: ByteByReference,
    )

    fun document_editor_get_page_count(
        handle: Pointer,
        code: IntByReference,
    ): Int

    // ── DocumentEditor: metadata ──────────────────────────────────────────────
    fun document_editor_get_producer(
        handle: Pointer,
        code: IntByReference,
    ): Pointer?

    fun document_editor_set_producer(
        handle: Pointer,
        value: String,
        code: IntByReference,
    ): Int

    fun document_editor_get_creation_date(
        handle: Pointer,
        code: IntByReference,
    ): Pointer?

    fun document_editor_set_creation_date(
        handle: Pointer,
        dateStr: String,
        code: IntByReference,
    ): Int

    // ── DocumentEditor: save / export ─────────────────────────────────────────
    fun document_editor_save(
        handle: Pointer,
        path: String,
        code: IntByReference,
    ): Int

    fun document_editor_save_to_bytes(
        handle: Pointer,
        outLen: LongByReference,
        code: IntByReference,
    ): Pointer?

    fun document_editor_save_to_bytes_with_options(
        handle: Pointer,
        compress: Boolean,
        garbageCollect: Boolean,
        linearize: Boolean,
        outLen: LongByReference,
        code: IntByReference,
    ): Pointer?

    fun document_editor_extract_pages_to_bytes(
        handle: Pointer,
        pages: IntArray,
        count: Long,
        outLen: LongByReference,
        code: IntByReference,
    ): Pointer?

    fun document_editor_convert_to_pdf_a(
        handle: Pointer,
        level: Int,
        code: IntByReference,
    ): Int

    fun document_editor_save_encrypted_to_bytes(
        handle: Pointer,
        userPassword: String,
        ownerPassword: String,
        outLen: LongByReference,
        code: IntByReference,
    ): Pointer?

    fun document_editor_save_encrypted(
        handle: Pointer,
        path: String,
        userPassword: String,
        ownerPassword: String,
        code: IntByReference,
    ): Int

    // ── DocumentEditor: merge / embed ─────────────────────────────────────────
    fun document_editor_merge_from(
        handle: Pointer,
        sourcePath: String,
        code: IntByReference,
    ): Int

    fun document_editor_merge_from_bytes(
        handle: Pointer,
        data: ByteArray,
        len: Long,
        code: IntByReference,
    ): Int

    fun document_editor_embed_file(
        handle: Pointer,
        name: String,
        data: ByteArray,
        len: Long,
        code: IntByReference,
    ): Int

    // ── DocumentEditor: pages ─────────────────────────────────────────────────
    fun document_editor_delete_page(
        handle: Pointer,
        pageIndex: Int,
        code: IntByReference,
    ): Int

    fun document_editor_move_page(
        handle: Pointer,
        from: Int,
        to: Int,
        code: IntByReference,
    ): Int

    // ── DocumentEditor: rotation ──────────────────────────────────────────────
    fun document_editor_rotate_all_pages(
        handle: Pointer,
        degrees: Int,
        code: IntByReference,
    ): Int

    fun document_editor_rotate_page_by(
        handle: Pointer,
        page: Long,
        degrees: Int,
        code: IntByReference,
    ): Int

    fun document_editor_get_page_rotation(
        handle: Pointer,
        page: Int,
        code: IntByReference,
    ): Int

    fun document_editor_set_page_rotation(
        handle: Pointer,
        page: Int,
        degrees: Int,
        code: IntByReference,
    ): Int

    // ── DocumentEditor: geometry ──────────────────────────────────────────────
    fun document_editor_crop_margins(
        handle: Pointer,
        left: Float,
        right: Float,
        top: Float,
        bottom: Float,
        code: IntByReference,
    ): Int

    fun document_editor_get_page_media_box(
        handle: Pointer,
        page: Long,
        x: DoubleByReference,
        y: DoubleByReference,
        w: DoubleByReference,
        h: DoubleByReference,
        code: IntByReference,
    ): Int

    fun document_editor_set_page_media_box(
        handle: Pointer,
        page: Long,
        x: Double,
        y: Double,
        w: Double,
        h: Double,
        code: IntByReference,
    ): Int

    fun document_editor_get_page_crop_box(
        handle: Pointer,
        page: Long,
        x: DoubleByReference,
        y: DoubleByReference,
        w: DoubleByReference,
        h: DoubleByReference,
        code: IntByReference,
    ): Int

    fun document_editor_set_page_crop_box(
        handle: Pointer,
        page: Long,
        x: Double,
        y: Double,
        w: Double,
        h: Double,
        code: IntByReference,
    ): Int

    // ── DocumentEditor: erase regions ─────────────────────────────────────────
    fun document_editor_erase_region(
        handle: Pointer,
        page: Int,
        x: Float,
        y: Float,
        w: Float,
        h: Float,
        code: IntByReference,
    ): Int

    fun document_editor_erase_regions(
        handle: Pointer,
        page: Long,
        rects: DoubleArray,
        rectsCount: Long,
        code: IntByReference,
    ): Int

    fun document_editor_clear_erase_regions(
        handle: Pointer,
        page: Long,
        code: IntByReference,
    ): Int

    // ── DocumentEditor: redaction ─────────────────────────────────────────────
    fun document_editor_apply_page_redactions(
        handle: Pointer,
        page: Long,
        code: IntByReference,
    ): Int

    fun document_editor_apply_all_redactions(
        handle: Pointer,
        code: IntByReference,
    ): Int

    fun document_editor_is_page_marked_for_redaction(
        handle: Pointer,
        page: Long,
    ): Int

    fun document_editor_unmark_page_for_redaction(
        handle: Pointer,
        page: Long,
        code: IntByReference,
    ): Int

    // ── DocumentEditor: flatten ───────────────────────────────────────────────
    fun document_editor_flatten_annotations(
        handle: Pointer,
        page: Int,
        code: IntByReference,
    ): Int

    fun document_editor_flatten_all_annotations(
        handle: Pointer,
        code: IntByReference,
    ): Int

    fun document_editor_flatten_forms(
        handle: Pointer,
        code: IntByReference,
    ): Int

    fun document_editor_flatten_forms_on_page(
        handle: Pointer,
        pageIndex: Int,
        code: IntByReference,
    ): Int

    fun document_editor_flatten_warnings_count(handle: Pointer): Int

    fun document_editor_flatten_warning(
        handle: Pointer,
        index: Int,
        code: IntByReference,
    ): Pointer?

    fun document_editor_is_page_marked_for_flatten(
        handle: Pointer,
        page: Long,
    ): Int

    fun document_editor_unmark_page_for_flatten(
        handle: Pointer,
        page: Long,
        code: IntByReference,
    ): Int

    // ── DocumentEditor: forms ─────────────────────────────────────────────────
    fun document_editor_set_form_field_value(
        handle: Pointer,
        name: String,
        value: String,
        code: IntByReference,
    ): Int
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

    /**
     * Copy an owned uint8 buffer of [len] bytes into a Kotlin [ByteArray] and
     * free it via free_bytes. A null pointer is treated as a failure.
     */
    fun takeBytes(
        p: Pointer?,
        len: Long,
        code: Int,
        op: String,
    ): ByteArray {
        if (p == null) throw PdfOxideException(code, op)
        val n = if (len < 0) 0 else len.toInt()
        val out = p.getByteArray(0, n)
        lib.free_bytes(p)
        return out
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

/**
 * A PDF opened for in-place editing. Wraps every `document_editor_*` C function
 * over an owned native handle, freed on [close] (AutoCloseable) with a finalizer
 * backstop. Page indices are 0-based. Mutating calls raise [PdfOxideException]
 * on a non-success status code or a set error code; the `is*` queries return a
 * Boolean (1 = true). Close when done.
 */
class DocumentEditor internal constructor(
    private var handle: Pointer?,
) : AutoCloseable {
    private fun ptr(): Pointer = handle ?: error("DocumentEditor is closed")

    companion object {
        @JvmStatic
        fun openEditor(path: String): DocumentEditor {
            val code = IntByReference()
            val h =
                Native_.lib.document_editor_open(path, code)
                    ?: throw PdfOxideException(code.value, "openEditor")
            return DocumentEditor(h)
        }

        /** Alias for [openEditor]. */
        @JvmStatic
        fun open(path: String): DocumentEditor = openEditor(path)

        @JvmStatic
        fun openFromBytes(data: ByteArray): DocumentEditor {
            val code = IntByReference()
            val h =
                Native_.lib.document_editor_open_from_bytes(data, data.size.toLong(), code)
                    ?: throw PdfOxideException(code.value, "openFromBytes")
            return DocumentEditor(h)
        }
    }

    // ── Inspection ────────────────────────────────────────────────────────────
    fun pageCount(): Int {
        val code = IntByReference()
        val n = Native_.lib.document_editor_get_page_count(ptr(), code)
        if (n < 0 || code.value != 0) throw PdfOxideException(code.value, "pageCount")
        return n
    }

    fun version(): PdfVersion {
        val maj = ByteByReference()
        val min = ByteByReference()
        Native_.lib.document_editor_get_version(ptr(), maj, min)
        return PdfVersion(maj.value.toInt() and 0xFF, min.value.toInt() and 0xFF)
    }

    fun isModified(): Boolean = Native_.lib.document_editor_is_modified(ptr())

    fun getSourcePath(): String {
        val code = IntByReference()
        return Native_.readString(Native_.lib.document_editor_get_source_path(ptr(), code), code, "getSourcePath")
    }

    // ── Metadata ──────────────────────────────────────────────────────────────
    fun getProducer(): String {
        val code = IntByReference()
        return Native_.readString(Native_.lib.document_editor_get_producer(ptr(), code), code, "getProducer")
    }

    fun setProducer(value: String) {
        val code = IntByReference()
        if (Native_.lib.document_editor_set_producer(ptr(), value, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "setProducer")
        }
    }

    fun getCreationDate(): String {
        val code = IntByReference()
        return Native_.readString(Native_.lib.document_editor_get_creation_date(ptr(), code), code, "getCreationDate")
    }

    fun setCreationDate(dateStr: String) {
        val code = IntByReference()
        if (Native_.lib.document_editor_set_creation_date(ptr(), dateStr, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "setCreationDate")
        }
    }

    // ── Save / export ─────────────────────────────────────────────────────────
    fun save(path: String) {
        val code = IntByReference()
        if (Native_.lib.document_editor_save(ptr(), path, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "save")
        }
    }

    fun saveToBytes(): ByteArray {
        val len = LongByReference()
        val code = IntByReference()
        return Native_.takeBytes(Native_.lib.document_editor_save_to_bytes(ptr(), len, code), len.value, code.value, "saveToBytes")
    }

    fun saveToBytesWithOptions(
        compress: Boolean,
        garbageCollect: Boolean,
        linearize: Boolean,
    ): ByteArray {
        val len = LongByReference()
        val code = IntByReference()
        return Native_.takeBytes(
            Native_.lib.document_editor_save_to_bytes_with_options(ptr(), compress, garbageCollect, linearize, len, code),
            len.value,
            code.value,
            "saveToBytesWithOptions",
        )
    }

    fun extractPagesToBytes(pages: IntArray): ByteArray {
        val len = LongByReference()
        val code = IntByReference()
        return Native_.takeBytes(
            Native_.lib.document_editor_extract_pages_to_bytes(ptr(), pages, pages.size.toLong(), len, code),
            len.value,
            code.value,
            "extractPagesToBytes",
        )
    }

    fun convertToPdfA(level: Int) {
        val code = IntByReference()
        if (Native_.lib.document_editor_convert_to_pdf_a(ptr(), level, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "convertToPdfA")
        }
    }

    fun saveEncryptedToBytes(
        userPassword: String,
        ownerPassword: String,
    ): ByteArray {
        val len = LongByReference()
        val code = IntByReference()
        return Native_.takeBytes(
            Native_.lib.document_editor_save_encrypted_to_bytes(ptr(), userPassword, ownerPassword, len, code),
            len.value,
            code.value,
            "saveEncryptedToBytes",
        )
    }

    fun saveEncrypted(
        path: String,
        userPassword: String,
        ownerPassword: String,
    ) {
        val code = IntByReference()
        if (Native_.lib.document_editor_save_encrypted(ptr(), path, userPassword, ownerPassword, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "saveEncrypted")
        }
    }

    // ── Merge / embed ─────────────────────────────────────────────────────────
    fun mergeFrom(sourcePath: String) {
        val code = IntByReference()
        if (Native_.lib.document_editor_merge_from(ptr(), sourcePath, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "mergeFrom")
        }
    }

    fun mergeFromBytes(data: ByteArray) {
        val code = IntByReference()
        if (Native_.lib.document_editor_merge_from_bytes(ptr(), data, data.size.toLong(), code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "mergeFromBytes")
        }
    }

    fun embedFile(
        name: String,
        data: ByteArray,
    ) {
        val code = IntByReference()
        if (Native_.lib.document_editor_embed_file(ptr(), name, data, data.size.toLong(), code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "embedFile")
        }
    }

    // ── Pages ─────────────────────────────────────────────────────────────────
    fun deletePage(pageIndex: Int) {
        val code = IntByReference()
        if (Native_.lib.document_editor_delete_page(ptr(), pageIndex, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "deletePage")
        }
    }

    fun movePage(
        from: Int,
        to: Int,
    ) {
        val code = IntByReference()
        if (Native_.lib.document_editor_move_page(ptr(), from, to, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "movePage")
        }
    }

    // ── Rotation ──────────────────────────────────────────────────────────────
    fun rotateAllPages(degrees: Int) {
        val code = IntByReference()
        if (Native_.lib.document_editor_rotate_all_pages(ptr(), degrees, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "rotateAllPages")
        }
    }

    fun rotatePageBy(
        page: Int,
        degrees: Int,
    ) {
        val code = IntByReference()
        if (Native_.lib.document_editor_rotate_page_by(ptr(), page.toLong(), degrees, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "rotatePageBy")
        }
    }

    fun getPageRotation(page: Int): Int {
        val code = IntByReference()
        val deg = Native_.lib.document_editor_get_page_rotation(ptr(), page, code)
        if (code.value != 0) throw PdfOxideException(code.value, "getPageRotation")
        return deg
    }

    fun setPageRotation(
        page: Int,
        degrees: Int,
    ) {
        val code = IntByReference()
        if (Native_.lib.document_editor_set_page_rotation(ptr(), page, degrees, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "setPageRotation")
        }
    }

    // ── Geometry ──────────────────────────────────────────────────────────────
    fun cropMargins(
        left: Float,
        right: Float,
        top: Float,
        bottom: Float,
    ) {
        val code = IntByReference()
        if (Native_.lib.document_editor_crop_margins(ptr(), left, right, top, bottom, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "cropMargins")
        }
    }

    fun getPageMediaBox(page: Int): Bbox =
        readDoubleBbox(page, "getPageMediaBox") { p, x, y, w, h, c ->
            Native_.lib.document_editor_get_page_media_box(ptr(), p, x, y, w, h, c)
        }

    fun setPageMediaBox(
        page: Int,
        box: Bbox,
    ) {
        val code = IntByReference()
        val status =
            Native_.lib.document_editor_set_page_media_box(
                ptr(),
                page.toLong(),
                box.x.toDouble(),
                box.y.toDouble(),
                box.width.toDouble(),
                box.height.toDouble(),
                code,
            )
        if (status != 0 || code.value != 0) throw PdfOxideException(code.value, "setPageMediaBox")
    }

    fun getPageCropBox(page: Int): Bbox =
        readDoubleBbox(page, "getPageCropBox") { p, x, y, w, h, c ->
            Native_.lib.document_editor_get_page_crop_box(ptr(), p, x, y, w, h, c)
        }

    fun setPageCropBox(
        page: Int,
        box: Bbox,
    ) {
        val code = IntByReference()
        val status =
            Native_.lib.document_editor_set_page_crop_box(
                ptr(),
                page.toLong(),
                box.x.toDouble(),
                box.y.toDouble(),
                box.width.toDouble(),
                box.height.toDouble(),
                code,
            )
        if (status != 0 || code.value != 0) throw PdfOxideException(code.value, "setPageCropBox")
    }

    // ── Erase regions ─────────────────────────────────────────────────────────
    fun eraseRegion(
        page: Int,
        x: Float,
        y: Float,
        w: Float,
        h: Float,
    ) {
        val code = IntByReference()
        if (Native_.lib.document_editor_erase_region(ptr(), page, x, y, w, h, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "eraseRegion")
        }
    }

    /** Erase the given [rects] (each an (x, y, w, h) [Bbox]) on [page]. */
    fun eraseRegions(
        page: Int,
        rects: List<Bbox>,
    ) {
        val flat = DoubleArray(rects.size * 4)
        for ((i, r) in rects.withIndex()) {
            flat[i * 4] = r.x.toDouble()
            flat[i * 4 + 1] = r.y.toDouble()
            flat[i * 4 + 2] = r.width.toDouble()
            flat[i * 4 + 3] = r.height.toDouble()
        }
        val code = IntByReference()
        if (Native_.lib.document_editor_erase_regions(ptr(), page.toLong(), flat, rects.size.toLong(), code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "eraseRegions")
        }
    }

    fun clearEraseRegions(page: Int) {
        val code = IntByReference()
        if (Native_.lib.document_editor_clear_erase_regions(ptr(), page.toLong(), code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "clearEraseRegions")
        }
    }

    // ── Redaction ─────────────────────────────────────────────────────────────
    fun applyPageRedactions(page: Int) {
        val code = IntByReference()
        if (Native_.lib.document_editor_apply_page_redactions(ptr(), page.toLong(), code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "applyPageRedactions")
        }
    }

    fun applyAllRedactions() {
        val code = IntByReference()
        if (Native_.lib.document_editor_apply_all_redactions(ptr(), code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "applyAllRedactions")
        }
    }

    fun isPageMarkedForRedaction(page: Int): Boolean {
        val r = Native_.lib.document_editor_is_page_marked_for_redaction(ptr(), page.toLong())
        if (r < 0) throw PdfOxideException(r, "isPageMarkedForRedaction")
        return r == 1
    }

    fun unmarkPageForRedaction(page: Int) {
        val code = IntByReference()
        if (Native_.lib.document_editor_unmark_page_for_redaction(ptr(), page.toLong(), code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "unmarkPageForRedaction")
        }
    }

    // ── Flatten ───────────────────────────────────────────────────────────────
    fun flattenAnnotations(page: Int) {
        val code = IntByReference()
        if (Native_.lib.document_editor_flatten_annotations(ptr(), page, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "flattenAnnotations")
        }
    }

    fun flattenAllAnnotations() {
        val code = IntByReference()
        if (Native_.lib.document_editor_flatten_all_annotations(ptr(), code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "flattenAllAnnotations")
        }
    }

    fun flattenForms() {
        val code = IntByReference()
        if (Native_.lib.document_editor_flatten_forms(ptr(), code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "flattenForms")
        }
    }

    fun flattenFormsOnPage(pageIndex: Int) {
        val code = IntByReference()
        if (Native_.lib.document_editor_flatten_forms_on_page(ptr(), pageIndex, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "flattenFormsOnPage")
        }
    }

    fun flattenWarningsCount(): Int {
        val n = Native_.lib.document_editor_flatten_warnings_count(ptr())
        if (n < 0) throw PdfOxideException(n, "flattenWarningsCount")
        return n
    }

    fun flattenWarning(index: Int): String {
        val code = IntByReference()
        return Native_.readString(Native_.lib.document_editor_flatten_warning(ptr(), index, code), code, "flattenWarning")
    }

    fun isPageMarkedForFlatten(page: Int): Boolean {
        val r = Native_.lib.document_editor_is_page_marked_for_flatten(ptr(), page.toLong())
        if (r < 0) throw PdfOxideException(r, "isPageMarkedForFlatten")
        return r == 1
    }

    fun unmarkPageForFlatten(page: Int) {
        val code = IntByReference()
        if (Native_.lib.document_editor_unmark_page_for_flatten(ptr(), page.toLong(), code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "unmarkPageForFlatten")
        }
    }

    // ── Forms ─────────────────────────────────────────────────────────────────
    fun setFormFieldValue(
        name: String,
        value: String,
    ) {
        val code = IntByReference()
        if (Native_.lib.document_editor_set_form_field_value(ptr(), name, value, code) != 0 || code.value != 0) {
            throw PdfOxideException(code.value, "setFormFieldValue")
        }
    }

    private inline fun readDoubleBbox(
        page: Int,
        op: String,
        getter: (Long, DoubleByReference, DoubleByReference, DoubleByReference, DoubleByReference, IntByReference) -> Int,
    ): Bbox {
        val x = DoubleByReference()
        val y = DoubleByReference()
        val w = DoubleByReference()
        val h = DoubleByReference()
        val code = IntByReference()
        if (getter(page.toLong(), x, y, w, h, code) != 0 || code.value != 0) throw PdfOxideException(code.value, op)
        return Bbox(x.value.toFloat(), y.value.toFloat(), w.value.toFloat(), h.value.toFloat())
    }

    override fun close() {
        handle?.let {
            Native_.lib.document_editor_free(it)
            handle = null
        }
    }

    @Suppress("ProtectedInFinal")
    protected fun finalize() {
        close()
    }
}
