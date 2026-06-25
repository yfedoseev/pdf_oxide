// html_extraction — build a PDF from Markdown, open it, render HTML.
//
// Shared regression scenario (mirrored across language bindings). Exits
// non-zero on any failed assertion; prints "HTML OK" on success.
import 'dart:io';

import 'package:pdf_oxide/pdf_oxide.dart';

void main() {
  final pdf = Pdf.fromMarkdown(
      '# Hello pdf_oxide\n\nThis is a **Dart** regression example.\n');
  final doc = PdfDocument.openFromBytes(pdf.toBytes());
  try {
    final html = doc.toHtmlAll();
    print('--- html (all) ---');
    print(html);

    if (!html.contains('<')) {
      stderr.writeln("assertion failed: html does not contain '<'");
      exit(1);
    }
    if (!html.contains('pdf_oxide')) {
      stderr.writeln("assertion failed: html does not contain 'pdf_oxide'");
      exit(1);
    }

    print('HTML OK');
  } finally {
    doc.close();
    pdf.close();
  }
}
