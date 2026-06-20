// pdf_oxide — Scala 3 idiomatic facade.
//
// A THIN wrapper over the mature `fyi.oxide:pdf-oxide` Java binding (which owns
// the single JNI native bridge). The Java types (PdfDocument, Pdf, PdfPage,
// DocumentEditor, PdfSigner, PdfValidator, AutoExtractor, and the
// geometry/text/table/search value types) are used directly from the dependency;
// `scala.util.Using` works out of the box on the AutoCloseable handles.
//
// These Scala 3 `extension` methods add the idioms a Scala caller wants over the
// Java surface: `java.util.Optional[T]` -> `Option[T]` and
// `java.util.List[T]` -> `Seq[T]`. No native code, no re-implementation.
package fyi.oxide.pdf

import fyi.oxide.pdf.annotation.Annotation
import fyi.oxide.pdf.auto.AutoResult
import fyi.oxide.pdf.compliance.ValidationViolation
import fyi.oxide.pdf.form.FormField
import fyi.oxide.pdf.image.ExtractedImage
import fyi.oxide.pdf.metadata.DocumentInfo
import fyi.oxide.pdf.search.{SearchMatch, SearchOptions}
import fyi.oxide.pdf.table.Table
import fyi.oxide.pdf.text.{TextChar, TextLine, TextWord}

import java.util.Optional
import scala.jdk.CollectionConverters.*

/** Generic `java.util.Optional[T]` -> Scala `Option[T]`. */
extension [T](o: Optional[T]) def toOption: Option[T] = if o.isPresent then Some(o.get) else None

/** PdfDocument: nullable metadata as `Option`, Java `List` returns as `Seq`. */
extension (doc: PdfDocument)
  def producerOption: Option[String] = doc.producer.toOption
  def creatorOption: Option[String] = doc.creator.toOption
  def formFieldsSeq: Seq[FormField] = doc.formFields.asScala.toSeq
  def pagesSeq: Seq[PdfPage] = doc.pages.asScala.toSeq
  def searchSeq(query: String): Seq[SearchMatch] = doc.search(query).asScala.toSeq

/** PdfPage: element extractors as Scala `Seq`. */
extension (page: PdfPage)
  def wordsSeq: Seq[TextWord] = page.words.asScala.toSeq
  def linesSeq: Seq[TextLine] = page.lines.asScala.toSeq
  def charsSeq: Seq[TextChar] = page.chars.asScala.toSeq
  def tablesSeq: Seq[Table] = page.tables.asScala.toSeq
  def imagesSeq: Seq[ExtractedImage] = page.images.asScala.toSeq
  def annotationsSeq: Seq[Annotation] = page.annotations.asScala.toSeq

/** FormField: nullable accessors as `Option`. */
extension (f: FormField)
  def valueOption: Option[String] = f.value.toOption
  def bboxOption: Option[fyi.oxide.pdf.geometry.BBox] = f.bbox.toOption

/** Annotation: nullable accessors as `Option`. */
extension (a: Annotation)
  def contentsOption: Option[String] = a.contents.toOption
  def uriOption: Option[String] = a.uri.toOption

/** AutoResult: optional renderings + page lists as `Option`/`Seq`. */
extension (r: AutoResult)
  def markdownOption: Option[String] = r.markdown.toOption
  def htmlOption: Option[String] = r.html.toOption
  def pagesNeedingOcrSeq: Seq[Int] = r.pagesNeedingOcr.asScala.toSeq.map(_.intValue)

/** ValidationViolation: optional page index as `Option[Int]`. */
extension (v: ValidationViolation)
  def pageIndexOption: Option[Int] = v.pageIndex.toOption.map(_.intValue)

/** SearchOptions: result cap as `Option[Int]`. */
extension (o: SearchOptions)
  def maxResultsOption: Option[Int] = o.maxResults.toOption.map(_.intValue)

/** DocumentInfo: every metadata field as `Option[String]`. */
extension (i: DocumentInfo)
  def titleOption: Option[String] = i.title.toOption
  def authorOption: Option[String] = i.author.toOption
  def subjectOption: Option[String] = i.subject.toOption
  def keywordsOption: Option[String] = i.keywords.toOption
  def creatorOption: Option[String] = i.creator.toOption
  def producerOption: Option[String] = i.producer.toOption
  def creationDateOption: Option[String] = i.creationDate.toOption
  def modificationDateOption: Option[String] = i.modificationDate.toOption
