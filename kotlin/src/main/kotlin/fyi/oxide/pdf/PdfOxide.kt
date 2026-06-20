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

    fun pdf_document_extract_structured_to_json(
        h: Pointer,
        page: Int,
        code: IntByReference,
    ): Pointer?

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
}

/** An opened PDF for extraction/inspection. Close when done (AutoCloseable). */
class PdfDocument internal constructor(
    private var handle: Pointer?,
) : AutoCloseable {
    private fun ptr(): Pointer = handle ?: throw IllegalStateException("PdfDocument is closed")

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

    fun extractStructuredJson(page: Int): String {
        val code = IntByReference()
        return Native_.takeString(
            Native_.lib.pdf_document_extract_structured_to_json(ptr(), page, code),
            code.value,
            "extractStructuredJson",
        )
    }

    override fun close() {
        handle?.let {
            Native_.lib.pdf_document_free(it)
            handle = null
        }
    }
}

/** A PDF produced by a builder. Close when done (AutoCloseable). */
class Pdf internal constructor(
    private var handle: Pointer?,
) : AutoCloseable {
    private fun ptr(): Pointer = handle ?: throw IllegalStateException("Pdf is closed")

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

    fun saveToBytes(): ByteArray {
        val len = IntByReference()
        val code = IntByReference()
        val p =
            Native_.lib.pdf_save_to_bytes(ptr(), len, code)
                ?: throw PdfOxideException(code.value, "saveToBytes")
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
