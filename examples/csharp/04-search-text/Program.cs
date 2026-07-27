// Search for a term across all pages of a PDF and print matches.
// Run: dotnet run -- document.pdf "query"

using PdfOxide.Core;

if (args.Length < 2)
{
    Console.Error.WriteLine("Usage: dotnet run -- <file.pdf> \"query\"");
    return 1;
}

var path = args[0];
var query = args[1];

using var doc = PdfDocument.Open(path);
var pages = doc.PageCount;
Console.WriteLine($"Searching for \"{query}\" in {path} ({pages} pages)...\n");

// Build the per-page search index for every page up front, instead of
// paying for it lazily on the first SearchPage()/SearchAll() call. Worth
// it here since we're about to search every page anyway.
doc.PrepareSearch();

var total = 0;
var pagesWithHits = 0;

for (int i = 0; i < pages; i++)
{
    var results = doc.SearchPage(i, query);
    if (results.Length == 0) continue;

    pagesWithHits++;
    Console.WriteLine($"Page {i + 1}: {results.Length} match(es)");
    foreach (var r in results)
    {
        Console.WriteLine($"  - \"{r.Text}\" (x={r.X:F1} y={r.Y:F1})");
        total++;
    }
    Console.WriteLine();
}

Console.WriteLine($"Found {total} total matches across {pagesWithHits} pages.");

// Free the cached search index now that we're done searching — useful
// before heavy extraction work on the same document object.
doc.ClearSearchIndex();
return 0;
