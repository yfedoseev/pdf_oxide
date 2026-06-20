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
  def pdf_from_markdown(md: String, code: IntByReference): Pointer
  def pdf_from_html(html: String, code: IntByReference): Pointer
  def pdf_from_text(text: String, code: IntByReference): Pointer
  def pdf_free(h: Pointer): Unit
  def pdf_save(h: Pointer, path: String, code: IntByReference): Int
  def pdf_save_to_bytes(h: Pointer, len: IntByReference, code: IntByReference): Pointer
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
