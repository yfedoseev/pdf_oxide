// pdf_oxide — Kotlin idiomatic facade.
//
// This module is a THIN wrapper over the mature `fyi.oxide:pdf-oxide` Java
// binding (which owns the single JNI native bridge). The Java types
// (PdfDocument, Pdf, PdfPage, DocumentEditor, PdfSigner, PdfValidator,
// AutoExtractor, the geometry/text/table/search value types, …) are re-exported
// transitively via the `api(...)` dependency, so Kotlin callers simply
// `import fyi.oxide.pdf.*` and use them directly — with `use { }` working out of
// the box on the AutoCloseable handles (Pdf, PdfDocument, DocumentEditor).
//
// The only thing a Kotlin caller needs beyond the Java API is null-safety over
// Java's `Optional<T>` returns. These extensions provide exactly that — no
// native code, no re-implementation of the API surface.
package fyi.oxide.pdf

import fyi.oxide.pdf.annotation.Annotation
import fyi.oxide.pdf.auto.AutoResult
import fyi.oxide.pdf.compliance.ValidationViolation
import fyi.oxide.pdf.form.FormField
import fyi.oxide.pdf.geometry.BBox
import fyi.oxide.pdf.metadata.DocumentInfo
import fyi.oxide.pdf.search.SearchOptions
import java.util.Optional

/** Kotlin-idiomatic view of a Java [Optional]: empty -> `null`. */
fun <T : Any> Optional<T>.orNull(): T? = orElse(null)

// ── PdfDocument metadata (Optional -> nullable) ─────────────────────────────

/** Document `/Producer`, or `null` if absent. */
fun PdfDocument.producerOrNull(): String? = producer().orElse(null)

/** Document `/Creator`, or `null` if absent. */
fun PdfDocument.creatorOrNull(): String? = creator().orElse(null)

// ── FormField (Optional -> nullable) ────────────────────────────────────────

/** Field value, or `null` if unset. */
fun FormField.valueOrNull(): String? = value().orElse(null)

/** Field widget bounding box, or `null` if the field has no widget. */
fun FormField.bboxOrNull(): BBox? = bbox().orElse(null)

// ── Annotation (Optional -> nullable) ───────────────────────────────────────

/** Annotation `/Contents`, or `null`. */
fun Annotation.contentsOrNull(): String? = contents().orElse(null)

/** Link annotation target URI, or `null` for non-link annotations. */
fun Annotation.uriOrNull(): String? = uri().orElse(null)

// ── AutoResult (Optional -> nullable) ───────────────────────────────────────

/** Markdown rendering of the auto-extraction, or `null` if not produced. */
fun AutoResult.markdownOrNull(): String? = markdown().orElse(null)

/** HTML rendering of the auto-extraction, or `null` if not produced. */
fun AutoResult.htmlOrNull(): String? = html().orElse(null)

// ── ValidationViolation (Optional -> nullable) ──────────────────────────────

/** Page index the violation applies to, or `null` for document-level rules. */
fun ValidationViolation.pageIndexOrNull(): Int? = pageIndex().orElse(null)

// ── SearchOptions (Optional -> nullable) ────────────────────────────────────

/** Result cap, or `null` when unbounded. */
fun SearchOptions.maxResultsOrNull(): Int? = maxResults().orElse(null)

// ── DocumentInfo (Optional -> nullable, all metadata fields) ────────────────

/** Title, or `null`. */
fun DocumentInfo.titleOrNull(): String? = title().orElse(null)

/** Author, or `null`. */
fun DocumentInfo.authorOrNull(): String? = author().orElse(null)

/** Subject, or `null`. */
fun DocumentInfo.subjectOrNull(): String? = subject().orElse(null)

/** Keywords, or `null`. */
fun DocumentInfo.keywordsOrNull(): String? = keywords().orElse(null)

/** Creator, or `null`. */
fun DocumentInfo.creatorOrNull(): String? = creator().orElse(null)

/** Producer, or `null`. */
fun DocumentInfo.producerOrNull(): String? = producer().orElse(null)

/** Creation date, or `null`. */
fun DocumentInfo.creationDateOrNull(): String? = creationDate().orElse(null)

/** Modification date, or `null`. */
fun DocumentInfo.modificationDateOrNull(): String? = modificationDate().orElse(null)
