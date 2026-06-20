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
import com.sun.jna.ptr.{ByteByReference, IntByReference}

/** Thrown on any non-success C-ABI error code. */
final class PdfOxideException(val code: Int, val op: String)
    extends RuntimeException(s"pdf_oxide: $op failed (error code $code)")

/** PDF version (e.g. 1.7). */
final case class PdfVersion(major: Int, minor: Int):
  override def toString: String = s"$major.$minor"

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
  def pdf_document_extract_structured_to_json(h: Pointer, page: Int, code: IntByReference): Pointer
  def pdf_from_markdown(md: String, code: IntByReference): Pointer
  def pdf_from_html(html: String, code: IntByReference): Pointer
  def pdf_from_text(text: String, code: IntByReference): Pointer
  def pdf_free(h: Pointer): Unit
  def pdf_save(h: Pointer, path: String, code: IntByReference): Int
  def pdf_save_to_bytes(h: Pointer, len: IntByReference, code: IntByReference): Pointer
  def free_string(p: Pointer): Unit

private[pdf] object Native_ {
  val lib: CLib = Native.load("pdf_oxide", classOf[CLib])

  def takeString(p: Pointer, code: Int, op: String): String = {
    if (p == null) throw PdfOxideException(code, op)
    val s = p.getString(0)
    lib.free_string(p)
    s
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

  def toMarkdownAll(): String =
    val code = IntByReference()
    Native_.takeString(
      Native_.lib.pdf_document_to_markdown_all(ptr, code),
      code.getValue,
      "toMarkdownAll"
    )

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

/** A PDF produced by a builder. AutoCloseable. */
final class Pdf private (private var handle: Pointer) extends AutoCloseable:
  private def ptr: Pointer =
    if handle == null then throw IllegalStateException("Pdf is closed") else handle

  def save(path: String): Unit =
    val code = IntByReference()
    if Native_.lib.pdf_save(ptr, path, code) != 0 then
      throw PdfOxideException(code.getValue, "save")

  def saveToBytes(): Array[Byte] =
    val len = IntByReference(); val code = IntByReference()
    val p = Native_.lib.pdf_save_to_bytes(ptr, len, code)
    if p == null then throw PdfOxideException(code.getValue, "saveToBytes")
    val n = if len.getValue < 0 then 0 else len.getValue
    val out = p.getByteArray(0, n)
    Native_.lib.free_string(p)
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
