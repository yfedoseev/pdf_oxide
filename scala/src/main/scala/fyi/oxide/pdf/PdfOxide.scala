// pdf_oxide — idiomatic Scala 3 bindings over the C ABI via JNA.
//
// JNA loads the cdylib (libpdf_oxide) by name. Handles are AutoCloseable;
// returned C strings/buffers are copied into Scala and freed via free_string;
// non-success C-ABI error codes throw PdfOxideException.
//
// API surface mirrors the other language bindings; coverage is asserted by
// ApiCoverageSpec (one test per public method).
package fyi.oxide.pdf

import com.sun.jna.{Library, Native, Pointer}
import com.sun.jna.ptr.{ByteByReference, FloatByReference, IntByReference}

/** Thrown on any non-success C-ABI error code. */
final class PdfOxideException(val code: Int, val op: String)
    extends RuntimeException(s"pdf_oxide: $op failed (error code $code)")

/** PDF version (e.g. 1.7). */
final case class PdfVersion(major: Int, minor: Int):
  override def toString: String = s"$major.$minor"

/** An axis-aligned bounding box in PDF user-space points. */
final case class Bbox(x: Float, y: Float, width: Float, height: Float)

/** A single extracted character (glyph). */
final case class Char(character: Int, bbox: Bbox, fontName: String, fontSize: Float)

/** A single extracted word. */
final case class Word(text: String, bbox: Bbox, fontName: String, fontSize: Float, bold: Boolean)

/** A single extracted line of text. */
final case class TextLine(text: String, bbox: Bbox, wordCount: Int)

/** A single extracted table. Cell text is read lazily via [[cell]]. */
final case class Table(
    rowCount: Int,
    colCount: Int,
    hasHeader: Boolean,
    private val cellFn: (Int, Int) => String
):
  /** Text of the cell at (row, col); both 0-based. */
  def cell(row: Int, col: Int): String = cellFn(row, col)

/** A single embedded font. */
final case class Font(
    name: String,
    `type`: String,
    encoding: String,
    embedded: Boolean,
    subset: Boolean
)

/** A single embedded image, with its raw (encoded) byte data. */
final case class Image(
    width: Int,
    height: Int,
    bitsPerComponent: Int,
    format: String,
    colorspace: String,
    data: Array[Byte]
)

/** A rendered raster image of a page, owning the underlying native handle.
  *
  * `width`, `height` and `data` (the encoded image bytes, e.g. PNG) are read eagerly at
  * construction; `data` is copied into the JVM and the native byte buffer freed via `free_bytes`.
  * The native FfiRenderedImage handle is retained so [[save]] can encode directly from native; it
  * is released by [[close]] (the type is AutoCloseable). Use it inside `scala.util.Using` or call
  * [[close]] when done to avoid leaking the handle.
  */
final class RenderedImage private[pdf] (private var handle: Pointer) extends AutoCloseable:
  private def ptr: Pointer =
    if handle == null then throw IllegalStateException("RenderedImage is closed") else handle

  /** Image width in pixels. */
  val width: Int =
    val code = IntByReference()
    val w = Native_.lib.pdf_get_rendered_image_width(ptr, code)
    if code.getValue != 0 then
      Native_.lib.pdf_rendered_image_free(handle); handle = null
      throw PdfOxideException(code.getValue, "renderedImageWidth")
    w

  /** Image height in pixels. */
  val height: Int =
    val code = IntByReference()
    val h = Native_.lib.pdf_get_rendered_image_height(ptr, code)
    if code.getValue != 0 then
      Native_.lib.pdf_rendered_image_free(handle); handle = null
      throw PdfOxideException(code.getValue, "renderedImageHeight")
    h

  /** The encoded image bytes (e.g. PNG), eagerly copied into the JVM. */
  val data: Array[Byte] =
    val len = IntByReference(); val code = IntByReference()
    val p = Native_.lib.pdf_get_rendered_image_data(ptr, len, code)
    if p == null then
      Native_.lib.pdf_rendered_image_free(handle); handle = null
      throw PdfOxideException(code.getValue, "renderedImageData")
    val n = if len.getValue < 0 then 0 else len.getValue
    val bytes = p.getByteArray(0, n)
    Native_.lib.free_bytes(p)
    bytes

  /** Save the rendered image to `path`, encoding from the live native handle. */
  def save(path: String): Unit =
    val code = IntByReference()
    if Native_.lib.pdf_save_rendered_image(ptr, path, code) != 0 then
      throw PdfOxideException(code.getValue, "saveRenderedImage")

  def close(): Unit =
    if handle != null then
      Native_.lib.pdf_rendered_image_free(handle)
      handle = null

/** A single page annotation. */
final case class Annotation(
    `type`: String,
    subtype: String,
    content: String,
    author: String,
    rect: Bbox,
    borderWidth: Float
)

/** A single extracted vector path. */
final case class Path(
    bbox: Bbox,
    strokeWidth: Float,
    hasStroke: Boolean,
    hasFill: Boolean,
    operationCount: Int
)

/** A single full-text search hit. */
final case class SearchResult(text: String, page: Int, bbox: Bbox)

/** Raw JNA binding to the pdf_oxide C ABI (internal). */
private[pdf] trait CLib extends Library:
  def pdf_document_open(path: String, code: IntByReference): Pointer
  def pdf_document_open_from_bytes(data: Array[Byte], len: Long, code: IntByReference): Pointer
  def pdf_document_open_with_password(path: String, pw: String, code: IntByReference): Pointer
  def pdf_document_free(h: Pointer): Unit
  def pdf_document_get_page_count(h: Pointer, code: IntByReference): Int
  def pdf_document_get_version(h: Pointer, major: ByteByReference, minor: ByteByReference): Unit
  def pdf_document_is_encrypted(h: Pointer): Boolean
  def pdf_document_has_structure_tree(h: Pointer): Boolean
  def pdf_document_extract_text(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_document_to_plain_text(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_document_to_markdown(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_document_to_html(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_document_to_markdown_all(h: Pointer, code: IntByReference): Pointer
  def pdf_document_to_html_all(h: Pointer, code: IntByReference): Pointer
  def pdf_document_to_plain_text_all(h: Pointer, code: IntByReference): Pointer
  def pdf_document_authenticate(h: Pointer, password: String, code: IntByReference): Boolean
  def pdf_document_extract_structured_to_json(h: Pointer, page: Int, code: IntByReference): Pointer
  // ── Phase-1 element extraction ───────────────────────────────────────────────
  def pdf_document_extract_chars(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_oxide_char_count(list: Pointer): Int
  def pdf_oxide_char_get_char(list: Pointer, index: Int, code: IntByReference): Int
  def pdf_oxide_char_get_bbox(
      list: Pointer,
      index: Int,
      x: com.sun.jna.ptr.FloatByReference,
      y: com.sun.jna.ptr.FloatByReference,
      w: com.sun.jna.ptr.FloatByReference,
      h: com.sun.jna.ptr.FloatByReference,
      code: IntByReference
  ): Unit
  def pdf_oxide_char_get_font_name(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_char_get_font_size(list: Pointer, index: Int, code: IntByReference): Float
  def pdf_oxide_char_list_free(list: Pointer): Unit
  def pdf_document_extract_words(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_oxide_word_count(list: Pointer): Int
  def pdf_oxide_word_get_text(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_word_get_bbox(
      list: Pointer,
      index: Int,
      x: com.sun.jna.ptr.FloatByReference,
      y: com.sun.jna.ptr.FloatByReference,
      w: com.sun.jna.ptr.FloatByReference,
      h: com.sun.jna.ptr.FloatByReference,
      code: IntByReference
  ): Unit
  def pdf_oxide_word_get_font_name(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_word_get_font_size(list: Pointer, index: Int, code: IntByReference): Float
  def pdf_oxide_word_is_bold(list: Pointer, index: Int, code: IntByReference): Boolean
  def pdf_oxide_word_list_free(list: Pointer): Unit
  def pdf_document_extract_text_lines(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_oxide_line_count(list: Pointer): Int
  def pdf_oxide_line_get_text(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_line_get_bbox(
      list: Pointer,
      index: Int,
      x: com.sun.jna.ptr.FloatByReference,
      y: com.sun.jna.ptr.FloatByReference,
      w: com.sun.jna.ptr.FloatByReference,
      h: com.sun.jna.ptr.FloatByReference,
      code: IntByReference
  ): Unit
  def pdf_oxide_line_get_word_count(list: Pointer, index: Int, code: IntByReference): Int
  def pdf_oxide_line_list_free(list: Pointer): Unit
  def pdf_document_extract_tables(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_oxide_table_count(list: Pointer): Int
  def pdf_oxide_table_get_row_count(list: Pointer, index: Int, code: IntByReference): Int
  def pdf_oxide_table_get_col_count(list: Pointer, index: Int, code: IntByReference): Int
  def pdf_oxide_table_get_cell_text(
      list: Pointer,
      tableIndex: Int,
      row: Int,
      col: Int,
      code: IntByReference
  ): Pointer
  def pdf_oxide_table_has_header(list: Pointer, index: Int, code: IntByReference): Boolean
  def pdf_oxide_table_list_free(list: Pointer): Unit
  // ── Phase-2 element extraction ───────────────────────────────────────────────
  def pdf_document_get_embedded_fonts(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_oxide_font_count(list: Pointer): Int
  def pdf_oxide_font_get_name(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_font_get_type(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_font_get_encoding(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_font_is_embedded(list: Pointer, index: Int, code: IntByReference): Boolean
  def pdf_oxide_font_is_subset(list: Pointer, index: Int, code: IntByReference): Boolean
  def pdf_oxide_font_list_free(list: Pointer): Unit
  def pdf_document_get_embedded_images(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_oxide_image_count(list: Pointer): Int
  def pdf_oxide_image_get_width(list: Pointer, index: Int, code: IntByReference): Int
  def pdf_oxide_image_get_height(list: Pointer, index: Int, code: IntByReference): Int
  def pdf_oxide_image_get_bits_per_component(list: Pointer, index: Int, code: IntByReference): Int
  def pdf_oxide_image_get_format(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_image_get_colorspace(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_image_get_data(
      list: Pointer,
      index: Int,
      dataLen: IntByReference,
      code: IntByReference
  ): Pointer
  def pdf_oxide_image_list_free(list: Pointer): Unit
  def pdf_document_get_page_annotations(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_oxide_annotation_count(list: Pointer): Int
  def pdf_oxide_annotation_get_type(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_annotation_get_subtype(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_annotation_get_content(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_annotation_get_author(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_annotation_get_rect(
      list: Pointer,
      index: Int,
      x: FloatByReference,
      y: FloatByReference,
      w: FloatByReference,
      h: FloatByReference,
      code: IntByReference
  ): Unit
  def pdf_oxide_annotation_get_border_width(list: Pointer, index: Int, code: IntByReference): Float
  def pdf_oxide_annotation_list_free(list: Pointer): Unit
  def pdf_document_extract_paths(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_oxide_path_count(list: Pointer): Int
  def pdf_oxide_path_get_bbox(
      list: Pointer,
      index: Int,
      x: FloatByReference,
      y: FloatByReference,
      w: FloatByReference,
      h: FloatByReference,
      code: IntByReference
  ): Unit
  def pdf_oxide_path_get_stroke_width(list: Pointer, index: Int, code: IntByReference): Float
  def pdf_oxide_path_has_stroke(list: Pointer, index: Int, code: IntByReference): Boolean
  def pdf_oxide_path_has_fill(list: Pointer, index: Int, code: IntByReference): Boolean
  def pdf_oxide_path_get_operation_count(list: Pointer, index: Int, code: IntByReference): Int
  def pdf_oxide_path_list_free(list: Pointer): Unit
  def pdf_document_search_page(
      h: Pointer,
      page: Int,
      term: String,
      caseSensitive: Boolean,
      code: IntByReference
  ): Pointer
  def pdf_document_search_all(
      h: Pointer,
      term: String,
      caseSensitive: Boolean,
      code: IntByReference
  ): Pointer
  def pdf_oxide_search_result_count(list: Pointer): Int
  def pdf_oxide_search_result_get_text(list: Pointer, index: Int, code: IntByReference): Pointer
  def pdf_oxide_search_result_get_page(list: Pointer, index: Int, code: IntByReference): Int
  def pdf_oxide_search_result_get_bbox(
      list: Pointer,
      index: Int,
      x: FloatByReference,
      y: FloatByReference,
      w: FloatByReference,
      h: FloatByReference,
      code: IntByReference
  ): Unit
  def pdf_oxide_search_result_free(list: Pointer): Unit
  def pdf_from_markdown(md: String, code: IntByReference): Pointer
  def pdf_from_html(html: String, code: IntByReference): Pointer
  def pdf_from_text(text: String, code: IntByReference): Pointer
  def pdf_free(h: Pointer): Unit
  def pdf_save(h: Pointer, path: String, code: IntByReference): Int
  def pdf_save_to_bytes(h: Pointer, len: IntByReference, code: IntByReference): Pointer
  // ── Phase-3 page rendering ───────────────────────────────────────────────────
  def pdf_render_page(h: Pointer, pageIndex: Int, format: Int, code: IntByReference): Pointer
  def pdf_render_page_zoom(
      h: Pointer,
      pageIndex: Int,
      zoom: Float,
      format: Int,
      code: IntByReference
  ): Pointer
  def pdf_render_page_thumbnail(
      h: Pointer,
      pageIndex: Int,
      size: Int,
      format: Int,
      code: IntByReference
  ): Pointer
  def pdf_get_rendered_image_width(img: Pointer, code: IntByReference): Int
  def pdf_get_rendered_image_height(img: Pointer, code: IntByReference): Int
  def pdf_get_rendered_image_data(
      img: Pointer,
      dataLen: IntByReference,
      code: IntByReference
  ): Pointer
  def pdf_save_rendered_image(img: Pointer, filePath: String, code: IntByReference): Int
  def pdf_rendered_image_free(img: Pointer): Unit
  def free_string(p: Pointer): Unit
  def free_bytes(p: Pointer): Unit

private[pdf] object Native_ {
  val lib: CLib = Native.load("pdf_oxide", classOf[CLib])

  def takeString(p: Pointer, code: Int, op: String): String = {
    if (p == null) throw PdfOxideException(code, op)
    val s = p.getString(0)
    lib.free_string(p)
    s
  }

  /** Reads an out-param bbox from a list accessor, raising on a non-success code. */
  def readBbox(
      fn: (
          Pointer,
          Int,
          FloatByReference,
          FloatByReference,
          FloatByReference,
          FloatByReference,
          IntByReference
      ) => Unit,
      list: Pointer,
      index: Int,
      op: String
  ): Bbox = {
    val x = FloatByReference(); val y = FloatByReference()
    val w = FloatByReference(); val h = FloatByReference()
    val code = IntByReference()
    fn(list, index, x, y, w, h, code)
    if (code.getValue != 0) throw PdfOxideException(code.getValue, op)
    Bbox(x.getValue, y.getValue, w.getValue, h.getValue)
  }
}

/** An opened PDF for extraction/inspection. AutoCloseable. */
final class PdfDocument private (private var handle: Pointer) extends AutoCloseable:
  private def ptr: Pointer =
    if handle == null then throw IllegalStateException("PdfDocument is closed") else handle

  def pageCount(): Int =
    val code = IntByReference()
    val n = Native_.lib.pdf_document_get_page_count(ptr, code)
    if n < 0 then throw PdfOxideException(code.getValue, "pageCount")
    n

  def version(): PdfVersion =
    val maj = ByteByReference(); val min = ByteByReference()
    Native_.lib.pdf_document_get_version(ptr, maj, min)
    PdfVersion(maj.getValue & 0xff, min.getValue & 0xff)

  def isEncrypted(): Boolean = Native_.lib.pdf_document_is_encrypted(ptr)
  def hasStructureTree(): Boolean = Native_.lib.pdf_document_has_structure_tree(ptr)

  private def strPage(
      fn: (Pointer, Int, IntByReference) => Pointer,
      page: Int,
      op: String
  ): String =
    val code = IntByReference()
    Native_.takeString(fn(ptr, page, code), code.getValue, op)

  def extractText(page: Int): String =
    strPage(Native_.lib.pdf_document_extract_text, page, "extractText")
  def toPlainText(page: Int): String =
    strPage(Native_.lib.pdf_document_to_plain_text, page, "toPlainText")
  def toMarkdown(page: Int): String =
    strPage(Native_.lib.pdf_document_to_markdown, page, "toMarkdown")
  def toHtml(page: Int): String = strPage(Native_.lib.pdf_document_to_html, page, "toHtml")
  def extractStructuredJson(page: Int): String =
    strPage(Native_.lib.pdf_document_extract_structured_to_json, page, "extractStructuredJson")

  /** Per-character extraction for a 0-based page. */
  def extractChars(pageIndex: Int): List[Char] =
    val code = IntByReference()
    val list = Native_.lib.pdf_document_extract_chars(ptr, pageIndex, code)
    if list == null then throw PdfOxideException(code.getValue, "extractChars")
    try
      val n = Native_.lib.pdf_oxide_char_count(list)
      (0 until n).map { i =>
        val c = IntByReference()
        val ch = Native_.lib.pdf_oxide_char_get_char(list, i, c)
        if c.getValue != 0 then throw PdfOxideException(c.getValue, "extractChars")
        val bbox = Native_.readBbox(Native_.lib.pdf_oxide_char_get_bbox, list, i, "extractChars")
        val fc = IntByReference()
        val fontName = Native_.takeString(
          Native_.lib.pdf_oxide_char_get_font_name(list, i, fc),
          fc.getValue,
          "extractChars"
        )
        val sc = IntByReference()
        val fontSize = Native_.lib.pdf_oxide_char_get_font_size(list, i, sc)
        if sc.getValue != 0 then throw PdfOxideException(sc.getValue, "extractChars")
        Char(ch, bbox, fontName, fontSize)
      }.toList
    finally Native_.lib.pdf_oxide_char_list_free(list)

  /** Per-word extraction for a 0-based page. */
  def extractWords(pageIndex: Int): List[Word] =
    val code = IntByReference()
    val list = Native_.lib.pdf_document_extract_words(ptr, pageIndex, code)
    if list == null then throw PdfOxideException(code.getValue, "extractWords")
    try
      val n = Native_.lib.pdf_oxide_word_count(list)
      (0 until n).map { i =>
        val tc = IntByReference()
        val text = Native_.takeString(
          Native_.lib.pdf_oxide_word_get_text(list, i, tc),
          tc.getValue,
          "extractWords"
        )
        val bbox = Native_.readBbox(Native_.lib.pdf_oxide_word_get_bbox, list, i, "extractWords")
        val fc = IntByReference()
        val fontName = Native_.takeString(
          Native_.lib.pdf_oxide_word_get_font_name(list, i, fc),
          fc.getValue,
          "extractWords"
        )
        val sc = IntByReference()
        val fontSize = Native_.lib.pdf_oxide_word_get_font_size(list, i, sc)
        if sc.getValue != 0 then throw PdfOxideException(sc.getValue, "extractWords")
        val bc = IntByReference()
        val bold = Native_.lib.pdf_oxide_word_is_bold(list, i, bc)
        if bc.getValue != 0 then throw PdfOxideException(bc.getValue, "extractWords")
        Word(text, bbox, fontName, fontSize, bold)
      }.toList
    finally Native_.lib.pdf_oxide_word_list_free(list)

  /** Per-line extraction for a 0-based page. */
  def extractTextLines(pageIndex: Int): List[TextLine] =
    val code = IntByReference()
    val list = Native_.lib.pdf_document_extract_text_lines(ptr, pageIndex, code)
    if list == null then throw PdfOxideException(code.getValue, "extractTextLines")
    try
      val n = Native_.lib.pdf_oxide_line_count(list)
      (0 until n).map { i =>
        val tc = IntByReference()
        val text = Native_.takeString(
          Native_.lib.pdf_oxide_line_get_text(list, i, tc),
          tc.getValue,
          "extractTextLines"
        )
        val bbox =
          Native_.readBbox(Native_.lib.pdf_oxide_line_get_bbox, list, i, "extractTextLines")
        val wc = IntByReference()
        val wordCount = Native_.lib.pdf_oxide_line_get_word_count(list, i, wc)
        if wc.getValue != 0 then throw PdfOxideException(wc.getValue, "extractTextLines")
        TextLine(text, bbox, wordCount)
      }.toList
    finally Native_.lib.pdf_oxide_line_list_free(list)

  /** Table extraction for a 0-based page.
    *
    * The returned [[Table]] cell accessors are eager-copied at extraction time; the underlying
    * native list is freed before this method returns.
    */
  def extractTables(pageIndex: Int): List[Table] =
    val code = IntByReference()
    val list = Native_.lib.pdf_document_extract_tables(ptr, pageIndex, code)
    if list == null then throw PdfOxideException(code.getValue, "extractTables")
    try
      val n = Native_.lib.pdf_oxide_table_count(list)
      (0 until n).map { i =>
        val rc = IntByReference()
        val rowCount = Native_.lib.pdf_oxide_table_get_row_count(list, i, rc)
        if rc.getValue != 0 then throw PdfOxideException(rc.getValue, "extractTables")
        val cc = IntByReference()
        val colCount = Native_.lib.pdf_oxide_table_get_col_count(list, i, cc)
        if cc.getValue != 0 then throw PdfOxideException(cc.getValue, "extractTables")
        val hc = IntByReference()
        val hasHeader = Native_.lib.pdf_oxide_table_has_header(list, i, hc)
        if hc.getValue != 0 then throw PdfOxideException(hc.getValue, "extractTables")
        // Eagerly copy cell text so the table outlives the native list.
        val cells: Map[(Int, Int), String] =
          (for r <- 0 until rowCount; c <- 0 until colCount yield
            val ec = IntByReference()
            val txt = Native_.takeString(
              Native_.lib.pdf_oxide_table_get_cell_text(list, i, r, c, ec),
              ec.getValue,
              "extractTables"
            )
            (r, c) -> txt
          ).toMap
        Table(rowCount, colCount, hasHeader, (r, c) => cells.getOrElse((r, c), ""))
      }.toList
    finally Native_.lib.pdf_oxide_table_list_free(list)

  /** Embedded fonts for a 0-based page. */
  def embeddedFonts(pageIndex: Int): List[Font] =
    val code = IntByReference()
    val list = Native_.lib.pdf_document_get_embedded_fonts(ptr, pageIndex, code)
    if list == null then throw PdfOxideException(code.getValue, "embeddedFonts")
    try
      val n = Native_.lib.pdf_oxide_font_count(list)
      (0 until n).map { i =>
        val nc = IntByReference()
        val name = Native_.takeString(
          Native_.lib.pdf_oxide_font_get_name(list, i, nc),
          nc.getValue,
          "embeddedFonts"
        )
        val tc = IntByReference()
        val typ = Native_.takeString(
          Native_.lib.pdf_oxide_font_get_type(list, i, tc),
          tc.getValue,
          "embeddedFonts"
        )
        val ec = IntByReference()
        val encoding = Native_.takeString(
          Native_.lib.pdf_oxide_font_get_encoding(list, i, ec),
          ec.getValue,
          "embeddedFonts"
        )
        val mc = IntByReference()
        val embedded = Native_.lib.pdf_oxide_font_is_embedded(list, i, mc)
        if mc.getValue != 0 then throw PdfOxideException(mc.getValue, "embeddedFonts")
        val sc = IntByReference()
        val subset = Native_.lib.pdf_oxide_font_is_subset(list, i, sc)
        if sc.getValue != 0 then throw PdfOxideException(sc.getValue, "embeddedFonts")
        Font(name, typ, encoding, embedded, subset)
      }.toList
    finally Native_.lib.pdf_oxide_font_list_free(list)

  /** Embedded images for a 0-based page. */
  def embeddedImages(pageIndex: Int): List[Image] =
    val code = IntByReference()
    val list = Native_.lib.pdf_document_get_embedded_images(ptr, pageIndex, code)
    if list == null then throw PdfOxideException(code.getValue, "embeddedImages")
    try
      val n = Native_.lib.pdf_oxide_image_count(list)
      (0 until n).map { i =>
        val wc = IntByReference()
        val width = Native_.lib.pdf_oxide_image_get_width(list, i, wc)
        if wc.getValue != 0 then throw PdfOxideException(wc.getValue, "embeddedImages")
        val hc = IntByReference()
        val height = Native_.lib.pdf_oxide_image_get_height(list, i, hc)
        if hc.getValue != 0 then throw PdfOxideException(hc.getValue, "embeddedImages")
        val bc = IntByReference()
        val bitsPerComponent = Native_.lib.pdf_oxide_image_get_bits_per_component(list, i, bc)
        if bc.getValue != 0 then throw PdfOxideException(bc.getValue, "embeddedImages")
        val fc = IntByReference()
        val format = Native_.takeString(
          Native_.lib.pdf_oxide_image_get_format(list, i, fc),
          fc.getValue,
          "embeddedImages"
        )
        val cc = IntByReference()
        val colorspace = Native_.takeString(
          Native_.lib.pdf_oxide_image_get_colorspace(list, i, cc),
          cc.getValue,
          "embeddedImages"
        )
        val len = IntByReference()
        val dc = IntByReference()
        val p = Native_.lib.pdf_oxide_image_get_data(list, i, len, dc)
        if p == null then throw PdfOxideException(dc.getValue, "embeddedImages")
        val dn = if len.getValue < 0 then 0 else len.getValue
        val data = p.getByteArray(0, dn)
        Native_.lib.free_bytes(p)
        Image(width, height, bitsPerComponent, format, colorspace, data)
      }.toList
    finally Native_.lib.pdf_oxide_image_list_free(list)

  /** Annotations for a 0-based page. */
  def pageAnnotations(pageIndex: Int): List[Annotation] =
    val code = IntByReference()
    val list = Native_.lib.pdf_document_get_page_annotations(ptr, pageIndex, code)
    if list == null then throw PdfOxideException(code.getValue, "pageAnnotations")
    try
      val n = Native_.lib.pdf_oxide_annotation_count(list)
      (0 until n).map { i =>
        val tc = IntByReference()
        val typ = Native_.takeString(
          Native_.lib.pdf_oxide_annotation_get_type(list, i, tc),
          tc.getValue,
          "pageAnnotations"
        )
        val sc = IntByReference()
        val subtype = Native_.takeString(
          Native_.lib.pdf_oxide_annotation_get_subtype(list, i, sc),
          sc.getValue,
          "pageAnnotations"
        )
        val cc = IntByReference()
        val content = Native_.takeString(
          Native_.lib.pdf_oxide_annotation_get_content(list, i, cc),
          cc.getValue,
          "pageAnnotations"
        )
        val ac = IntByReference()
        val author = Native_.takeString(
          Native_.lib.pdf_oxide_annotation_get_author(list, i, ac),
          ac.getValue,
          "pageAnnotations"
        )
        val rect =
          Native_.readBbox(Native_.lib.pdf_oxide_annotation_get_rect, list, i, "pageAnnotations")
        val bc = IntByReference()
        val borderWidth = Native_.lib.pdf_oxide_annotation_get_border_width(list, i, bc)
        if bc.getValue != 0 then throw PdfOxideException(bc.getValue, "pageAnnotations")
        Annotation(typ, subtype, content, author, rect, borderWidth)
      }.toList
    finally Native_.lib.pdf_oxide_annotation_list_free(list)

  /** Vector path extraction for a 0-based page. */
  def extractPaths(pageIndex: Int): List[Path] =
    val code = IntByReference()
    val list = Native_.lib.pdf_document_extract_paths(ptr, pageIndex, code)
    if list == null then throw PdfOxideException(code.getValue, "extractPaths")
    try
      val n = Native_.lib.pdf_oxide_path_count(list)
      (0 until n).map { i =>
        val bbox = Native_.readBbox(Native_.lib.pdf_oxide_path_get_bbox, list, i, "extractPaths")
        val wc = IntByReference()
        val strokeWidth = Native_.lib.pdf_oxide_path_get_stroke_width(list, i, wc)
        if wc.getValue != 0 then throw PdfOxideException(wc.getValue, "extractPaths")
        val sc = IntByReference()
        val hasStroke = Native_.lib.pdf_oxide_path_has_stroke(list, i, sc)
        if sc.getValue != 0 then throw PdfOxideException(sc.getValue, "extractPaths")
        val fc = IntByReference()
        val hasFill = Native_.lib.pdf_oxide_path_has_fill(list, i, fc)
        if fc.getValue != 0 then throw PdfOxideException(fc.getValue, "extractPaths")
        val oc = IntByReference()
        val operationCount = Native_.lib.pdf_oxide_path_get_operation_count(list, i, oc)
        if oc.getValue != 0 then throw PdfOxideException(oc.getValue, "extractPaths")
        Path(bbox, strokeWidth, hasStroke, hasFill, operationCount)
      }.toList
    finally Native_.lib.pdf_oxide_path_list_free(list)

  private def readSearchResults(list: Pointer, op: String): List[SearchResult] =
    try
      val n = Native_.lib.pdf_oxide_search_result_count(list)
      (0 until n).map { i =>
        val tc = IntByReference()
        val text = Native_.takeString(
          Native_.lib.pdf_oxide_search_result_get_text(list, i, tc),
          tc.getValue,
          op
        )
        val pc = IntByReference()
        val page = Native_.lib.pdf_oxide_search_result_get_page(list, i, pc)
        if pc.getValue != 0 then throw PdfOxideException(pc.getValue, op)
        val bbox = Native_.readBbox(Native_.lib.pdf_oxide_search_result_get_bbox, list, i, op)
        SearchResult(text, page, bbox)
      }.toList
    finally Native_.lib.pdf_oxide_search_result_free(list)

  /** Search a single 0-based page for `term`. */
  def search(pageIndex: Int, term: String, caseSensitive: Boolean): List[SearchResult] =
    val code = IntByReference()
    val list = Native_.lib.pdf_document_search_page(ptr, pageIndex, term, caseSensitive, code)
    if list == null then throw PdfOxideException(code.getValue, "search")
    readSearchResults(list, "search")

  /** Search the whole document for `term`. */
  def searchAll(term: String, caseSensitive: Boolean): List[SearchResult] =
    val code = IntByReference()
    val list = Native_.lib.pdf_document_search_all(ptr, term, caseSensitive, code)
    if list == null then throw PdfOxideException(code.getValue, "searchAll")
    readSearchResults(list, "searchAll")

  def toMarkdownAll(): String =
    val code = IntByReference()
    Native_.takeString(
      Native_.lib.pdf_document_to_markdown_all(ptr, code),
      code.getValue,
      "toMarkdownAll"
    )

  def toHtmlAll(): String =
    val code = IntByReference()
    Native_.takeString(
      Native_.lib.pdf_document_to_html_all(ptr, code),
      code.getValue,
      "toHtmlAll"
    )

  def toPlainTextAll(): String =
    val code = IntByReference()
    Native_.takeString(
      Native_.lib.pdf_document_to_plain_text_all(ptr, code),
      code.getValue,
      "toPlainTextAll"
    )

  /** Attempt to authenticate against an encrypted document.
    *
    * Returns true/false for a correct/incorrect password; a wrong password is not an error. Only a
    * non-success error code raises PdfOxideException.
    */
  def authenticate(password: String): Boolean =
    val code = IntByReference()
    val ok = Native_.lib.pdf_document_authenticate(ptr, password, code)
    if code.getValue != 0 then throw PdfOxideException(code.getValue, "authenticate")
    ok

  /** Render a 0-based page to a [[RenderedImage]]. `format` is 0=PNG (default), 1=JPEG. */
  def renderPage(pageIndex: Int, format: Int = 0): RenderedImage =
    val code = IntByReference()
    val img = Native_.lib.pdf_render_page(ptr, pageIndex, format, code)
    if img == null then throw PdfOxideException(code.getValue, "renderPage")
    RenderedImage(img)

  /** Render a 0-based page at the given `zoom` factor. `format` is 0=PNG (default), 1=JPEG. */
  def renderPageZoom(pageIndex: Int, zoom: Float, format: Int = 0): RenderedImage =
    val code = IntByReference()
    val img = Native_.lib.pdf_render_page_zoom(ptr, pageIndex, zoom, format, code)
    if img == null then throw PdfOxideException(code.getValue, "renderPageZoom")
    RenderedImage(img)

  /** Render a thumbnail of a 0-based page fitting `size` pixels. `format` is 0=PNG (default). */
  def renderPageThumbnail(pageIndex: Int, size: Int, format: Int = 0): RenderedImage =
    val code = IntByReference()
    val img = Native_.lib.pdf_render_page_thumbnail(ptr, pageIndex, size, format, code)
    if img == null then throw PdfOxideException(code.getValue, "renderPageThumbnail")
    RenderedImage(img)

  /** A lightweight view of a single (0-based) page; keeps its document alive. */
  def page(index: Int): PdfPage = PdfPage(this, index)

  def close(): Unit =
    if handle != null then
      Native_.lib.pdf_document_free(handle)
      handle = null

object PdfDocument:
  def open(path: String): PdfDocument =
    val code = IntByReference()
    val h = Native_.lib.pdf_document_open(path, code)
    if h == null then throw PdfOxideException(code.getValue, "open")
    PdfDocument(h)

  def openFromBytes(data: Array[Byte]): PdfDocument =
    val code = IntByReference()
    val h = Native_.lib.pdf_document_open_from_bytes(data, data.length.toLong, code)
    if h == null then throw PdfOxideException(code.getValue, "openFromBytes")
    PdfDocument(h)

  def openWithPassword(path: String, password: String): PdfDocument =
    val code = IntByReference()
    val h = Native_.lib.pdf_document_open_with_password(path, password, code)
    if h == null then throw PdfOxideException(code.getValue, "openWithPassword")
    PdfDocument(h)

/** A single page of a PdfDocument.
  *
  * Holds a strong reference to its document so it cannot outlive it; each method delegates to the
  * corresponding per-page PdfDocument method.
  */
final class PdfPage private[pdf] (private val doc: PdfDocument, val index: Int):
  def text(): String = doc.extractText(index)
  def markdown(): String = doc.toMarkdown(index)
  def html(): String = doc.toHtml(index)
  def plainText(): String = doc.toPlainText(index)

/** A PDF produced by a builder. AutoCloseable. */
final class Pdf private (private var handle: Pointer) extends AutoCloseable:
  private def ptr: Pointer =
    if handle == null then throw IllegalStateException("Pdf is closed") else handle

  def save(path: String): Unit =
    val code = IntByReference()
    if Native_.lib.pdf_save(ptr, path, code) != 0 then
      throw PdfOxideException(code.getValue, "save")

  def toBytes(): Array[Byte] =
    val len = IntByReference(); val code = IntByReference()
    val p = Native_.lib.pdf_save_to_bytes(ptr, len, code)
    if p == null then throw PdfOxideException(code.getValue, "toBytes")
    val n = if len.getValue < 0 then 0 else len.getValue
    val out = p.getByteArray(0, n)
    Native_.lib.free_bytes(p)
    out

  def close(): Unit =
    if handle != null then
      Native_.lib.pdf_free(handle)
      handle = null

object Pdf:
  private def build(fn: (String, IntByReference) => Pointer, in: String, op: String): Pdf =
    val code = IntByReference()
    val h = fn(in, code)
    if h == null then throw PdfOxideException(code.getValue, op)
    Pdf(h)

  def fromMarkdown(md: String): Pdf = build(Native_.lib.pdf_from_markdown, md, "fromMarkdown")
  def fromHtml(html: String): Pdf = build(Native_.lib.pdf_from_html, html, "fromHtml")
  def fromText(text: String): Pdf = build(Native_.lib.pdf_from_text, text, "fromText")
