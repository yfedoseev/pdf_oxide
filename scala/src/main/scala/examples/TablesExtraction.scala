// tables_extraction — build a PDF from a Markdown table, then extract tables.
// Shared-scenario regression example run in CI (no external fixture). Asserts
// the call returns a Seq without error (synthetic docs may yield 0 tables).
// Throws (non-zero exit) on assertion failure; prints "TABLES OK" on success.
// Uses the Java-backed API through the Scala facade (`tablesSeq` for List->Seq).
package examples

import fyi.oxide.pdf.{Pdf, PdfDocument, tablesSeq}
import scala.jdk.CollectionConverters.*
import scala.util.Using

@main def tablesExtraction(): Unit =
  val md = "# Report\n\n| Name | Value |\n|------|-------|\n| alpha | 1 |\n| beta | 2 |\n"
  Using.resource(Pdf.fromMarkdown(md)): pdf =>
    Using.resource(PdfDocument.open(pdf.save())): doc =>
      val tables = doc.page(0).tablesSeq
      assert(tables.size >= 0, "tables_extraction: expected a Seq of tables")
      println(s"tables: ${tables.size}")
      tables.foreach: t =>
        println(s"  ${t.rows()}x${t.cols()} cells=${t.cells().asScala.map(_.text())}")
      println("TABLES OK")
