// Coroutine-friendly extensions — pdf_oxide's idiomatic Kotlin value-add.
// Extraction is CPU-bound native work; these run it off the caller's thread on
// Dispatchers.Default so it composes cleanly with structured concurrency.
package fyi.oxide.pdf

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Suspending whole-document Markdown extraction. */
suspend fun PdfDocument.toMarkdownAllAsync(): String =
    withContext(Dispatchers.Default) { toMarkdownAll() }

/** Suspending per-page text extraction. */
suspend fun PdfDocument.extractTextAsync(page: Int): String =
    withContext(Dispatchers.Default) { extractText(page) }

/** Suspending per-page Markdown extraction. */
suspend fun PdfDocument.toMarkdownAsync(page: Int): String =
    withContext(Dispatchers.Default) { toMarkdown(page) }
