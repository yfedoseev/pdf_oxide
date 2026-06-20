// basic_extraction — build a PDF from Markdown, then extract it back.
// Run in CI as a smoke example (no external fixture).
import 'package:pdf_oxide/pdf_oxide.dart';

void main() {
  final pdf = Pdf.fromMarkdown(
      '# Hello pdf_oxide\n\nThis is a **Dart** binding smoke example.\n');
  final doc = PdfDocument.openFromBytes(pdf.toBytes());
  try {
    print('pages:   ${doc.pageCount}');
    print('version: ${doc.version}');
    print('--- text (page 0) ---');
    print(doc.extractText(0));
    print('--- markdown (all) ---');
    print(doc.toMarkdownAll());
  } finally {
    doc.close();
    pdf.close();
  }
}
