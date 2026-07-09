// Extract words with bounding boxes and tables from a PDF page.
// Run: dotnet run --project csharp/ -- document.pdf

using PdfOxide.Core;

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: dotnet run -- <file.pdf>");
    return 1;
}

var path = args[0];
using var doc = PdfDocument.Open(path);
Console.WriteLine($"Opened: {path}");

var page = 0;

// Words with position data
var words = doc.ExtractWords(page);
Console.WriteLine($"\n--- Words (page {page + 1}) ---");
foreach (var (text, x, y, width, height, sequence, rotationDegrees) in words.Take(20))
{
    Console.WriteLine($"{"\"" + text + "\"",-20} x={x,-7:F1} y={y,-7:F1} w={width,-7:F1} h={height,-7:F1} seq={sequence} rot={rotationDegrees}");
}
if (words.Length > 20)
    Console.WriteLine($"... ({words.Length - 20} more words)");

// Tables
var tables = doc.ExtractTables(page);
Console.WriteLine($"\n--- Tables (page {page + 1}) ---");
if (tables.Length == 0)
    Console.WriteLine("(no tables found)");
for (var i = 0; i < tables.Length; i++)
{
    var t = tables[i];
    Console.WriteLine($"Table {i + 1}: {t.RowCount} rows x {t.ColCount} cols");
    for (var r = 0; r < Math.Min(t.RowCount, 5); r++)
    {
        for (var c = 0; c < Math.Min(t.ColCount, 6); c++)
            Console.Write($"  [{r},{c}] \"{t.CellText(r, c)}\"");
        Console.WriteLine();
    }
}

return 0;
