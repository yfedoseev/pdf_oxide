// words_geometry — build a PDF from Markdown, open it, extract word geometry.
//
// Shared regression scenario (mirrored across language bindings). Exits
// non-zero on any failed assertion; prints "WORDS OK" on success.
import 'dart:io';

import 'package:pdf_oxide/pdf_oxide.dart';

void main() {
  final pdf = Pdf.fromMarkdown(
      '# Hello pdf_oxide\n\nThis is a **Dart** regression example.\n');
  final doc = PdfDocument.openFromBytes(pdf.toBytes());
  try {
    final words = doc.extractWords(0);
    print('word count: ${words.length}');

    if (words.isEmpty) {
      stderr.writeln('assertion failed: no words extracted');
      exit(1);
    }

    final first = words[0];
    print('first word: "${first.text}"  bbox=${first.bbox}');

    if (first.text != 'Hello') {
      stderr.writeln('assertion failed: first word is not "Hello"');
      exit(1);
    }
    if (!(first.bbox.width > 0.0 && first.bbox.height > 0.0)) {
      stderr.writeln('assertion failed: first word has no bbox');
      exit(1);
    }

    print('WORDS OK');
  } finally {
    doc.close();
    pdf.close();
  }
}
