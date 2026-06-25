// tables_extraction — build a PDF from Markdown with a table, extract tables.
//
// Shared regression scenario (mirrored across language bindings). The synthetic
// doc may yield zero tables; the contract is only that the call succeeds and
// returns a list (count >= 0). Exits non-zero on error; prints "TABLES OK".
import 'package:pdf_oxide/pdf_oxide.dart';

void main() {
  final pdf = Pdf.fromMarkdown(
      '# Report\n\n| Name | Value |\n|------|-------|\n| alpha | 1 |\n| beta | 2 |\n');
  final doc = PdfDocument.openFromBytes(pdf.toBytes());
  try {
    final tables = doc.extractTables(0);
    print('table count: ${tables.length}');

    for (var t = 0; t < tables.length; t++) {
      final tbl = tables[t];
      print('table $t: ${tbl.rowCount}x${tbl.colCount}');
      for (var r = 0; r < tbl.rowCount; r++) {
        for (var c = 0; c < tbl.colCount; c++) {
          print('  cell($r,$c)="${tbl.cell(r, c)}"');
        }
      }
    }

    // extractTables threw nothing -> the call returned a valid list.
    print('TABLES OK');
  } finally {
    doc.close();
    pdf.close();
  }
}
