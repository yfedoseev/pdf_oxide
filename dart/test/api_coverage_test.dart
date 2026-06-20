// One test per public method — mirrors the api_coverage convention used by
// every pdf_oxide binding. Self-contained: builds its own PDF from Markdown.
import 'dart:io';
import 'dart:typed_data';

import 'package:pdf_oxide/pdf_oxide.dart';
import 'package:test/test.dart';

Uint8List _samplePdf() => Pdf.fromMarkdown(
        '# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n')
    .toBytes();

void main() {
  group('Pdf builder', () {
    test('fromMarkdown + toBytes', () {
      final p = Pdf.fromMarkdown('# md\n\nbody\n');
      addTearDown(p.close);
      expect(p.toBytes().length, greaterThan(100));
    });
    test('fromHtml', () {
      final p = Pdf.fromHtml('<h1>html</h1><p>body</p>');
      addTearDown(p.close);
      expect(p.toBytes().length, greaterThan(100));
    });
    test('fromText', () {
      final p = Pdf.fromText('plain text body');
      addTearDown(p.close);
      expect(p.toBytes().length, greaterThan(100));
    });
    test('save', () {
      final p = Pdf.fromMarkdown('# f\n\nx\n');
      addTearDown(p.close);
      final path = '${Directory.systemTemp.path}/pdfoxide_dart_${pid}.pdf';
      p.save(path);
      expect(File(path).existsSync(), isTrue);
      File(path).deleteSync();
    });
  });

  group('PdfDocument', () {
    late PdfDocument doc;
    setUp(() => doc = PdfDocument.openFromBytes(_samplePdf()));
    tearDown(() => doc.close());

    test('openFromBytes + pageCount',
        () => expect(doc.pageCount, greaterThanOrEqualTo(1)));
    test('open (path)', () {
      final path = '${Directory.systemTemp.path}/pdfoxide_dart_open_${pid}.pdf';
      Pdf.fromMarkdown('# f\n\nx\n')
        ..save(path)
        ..close();
      final d = PdfDocument.open(path);
      addTearDown(() {
        d.close();
        File(path).deleteSync();
      });
      expect(d.pageCount, greaterThanOrEqualTo(1));
    });
    test('version', () => expect(doc.version.major, greaterThanOrEqualTo(1)));
    test('isEncrypted', () => expect(doc.isEncrypted(), isFalse));
    test('hasStructureTree', () => doc.hasStructureTree()); // smoke
    test('extractText', () => expect(doc.extractText(0), contains('Alpha')));
    test('toPlainText', () => expect(doc.toPlainText(0), isNotEmpty));
    test('toMarkdown', () => expect(doc.toMarkdown(0), isNotEmpty));
    test('toHtml', () => expect(doc.toHtml(0), contains('<')));
    test('toMarkdownAll', () => expect(doc.toMarkdownAll(), isNotEmpty));
    test('toHtmlAll', () => expect(doc.toHtmlAll(), contains('<')));
    test('toPlainTextAll', () => expect(doc.toPlainTextAll(), isNotEmpty));
    test('authenticate', () => expect(doc.authenticate('any'), isA<bool>()));
    test('extractStructuredJson',
        () => expect(doc.extractStructuredJson(0), isNotEmpty));
    test('page.text', () => expect(doc.page(0).text(), contains('Alpha')));
    test('page.markdown', () => expect(doc.page(0).markdown(), isNotEmpty));
    test('page.html', () => expect(doc.page(0).html(), contains('<')));
    test('page.plainText', () => expect(doc.page(0).plainText(), isNotEmpty));
  });

  test('error path: open nonexistent throws PdfOxideError', () {
    expect(() => PdfDocument.open('/nonexistent/nope.pdf'),
        throwsA(isA<PdfOxideError>()));
  });
}
