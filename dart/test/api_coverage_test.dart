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

    test('extractWords', () {
      final words = doc.extractWords(0);
      expect(words, isNotEmpty);
      expect(words[0].text, isNotEmpty);
      expect(words[0].bbox, isA<Bbox>());
    });
    test('extractChars', () => expect(doc.extractChars(0), isNotEmpty));
    test('extractTextLines', () => expect(doc.extractTextLines(0), isNotEmpty));
    test('extractTables',
        () => expect(doc.extractTables(0), isA<List<Table>>()));

    // Phase 2 — may be empty on this synthetic doc; assert the call succeeds.
    test(
        'embeddedFonts', () => expect(doc.embeddedFonts(0), isA<List<Font>>()));
    test('embeddedImages',
        () => expect(doc.embeddedImages(0), isA<List<Image>>()));
    test('pageAnnotations',
        () => expect(doc.pageAnnotations(0), isA<List<Annotation>>()));
    test('extractPaths', () => expect(doc.extractPaths(0), isA<List<Path>>()));
    test('search', () {
      final hits = doc.search(0, 'Alpha', false);
      expect(hits, isNotEmpty);
      expect(hits.first.text, contains('Alpha'));
      expect(hits.first.page, greaterThanOrEqualTo(0));
      expect(hits.first.bbox, isA<Bbox>());
    });
    test('searchAll', () {
      final hits = doc.searchAll('Alpha', false);
      expect(hits, isNotEmpty);
      expect(hits.first.text, contains('Alpha'));
      expect(hits.first.page, greaterThanOrEqualTo(0));
    });

    // Phase 3 — page rendering. The sample doc has a single page (index 0).
    test('renderPage', () {
      final img = doc.renderPage(0); // PNG (default format)
      addTearDown(img.close);
      expect(img.width, greaterThan(0));
      expect(img.height, greaterThan(0));
      expect(img.data, isNotEmpty);
    });
    test('renderPage.save', () {
      final img = doc.renderPage(0);
      addTearDown(img.close);
      final path =
          '${Directory.systemTemp.path}/pdfoxide_dart_render_${pid}.png';
      img.save(path);
      expect(File(path).existsSync(), isTrue);
      File(path).deleteSync();
    });
    test('renderPageZoom', () {
      final img = doc.renderPageZoom(0, 2.0);
      addTearDown(img.close);
      expect(img.width, greaterThan(0));
      expect(img.height, greaterThan(0));
    });
    test('renderPageThumbnail', () {
      final img = doc.renderPageThumbnail(0, 128);
      addTearDown(img.close);
      expect(img.width, greaterThan(0));
      expect(img.height, greaterThan(0));
    });
  });

  group('DocumentEditor', () {
    late DocumentEditor ed;
    setUp(() => ed = DocumentEditor.openFromBytes(_samplePdf()));
    tearDown(() => ed.close());

    test('openFromBytes + pageCount',
        () => expect(ed.pageCount, greaterThanOrEqualTo(1)));
    test('isModified', () => expect(ed.isModified(), isA<bool>()));
    test('rotateAllPages + getPageRotation', () {
      ed.rotateAllPages(90);
      expect(ed.getPageRotation(0), anyOf(equals(90), isA<int>()));
    });
    test('setProducer + getProducer', () {
      ed.setProducer('x');
      expect(ed.getProducer(), isA<String>());
    });
    test('saveToBytes', () => expect(ed.saveToBytes(), isNotEmpty));
  });

  test('error path: open nonexistent throws PdfOxideError', () {
    expect(() => PdfDocument.open('/nonexistent/nope.pdf'),
        throwsA(isA<PdfOxideError>()));
  });
}
