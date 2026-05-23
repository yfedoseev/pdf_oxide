<?php

declare(strict_types=1);

/**
 * Search and Extract Context Example
 *
 * Demonstrates how to search PDFs and extract surrounding context
 * around search results.
 */

require dirname(__DIR__) . '/vendor/autoload.php';

use PdfOxide\PdfDocument;
use PdfOxide\Builders\SearchOptions;

try {
    $pdfPath = $argv[1] ?? './sample.pdf';
    $searchTerm = $argv[2] ?? 'the';

    if (!file_exists($pdfPath)) {
        echo "Error: File not found: $pdfPath\n";
        echo "Usage: php 04_advanced_search.php <pdf-file> <search-term>\n";
        exit(1);
    }

    $pdf = new PdfDocument($pdfPath);

    echo "=== Advanced PDF Search Example ===\n";
    echo "File: $pdfPath\n";
    echo "Search term: '$searchTerm'\n\n";

    // Use SearchOptions for advanced filtering
    $options = (new SearchOptions())
        ->caseSensitive(false)
        ->wholeWordsOnly(false)
        ->maxResults(10);

    // Search document
    $results = $pdf->searchAll($searchTerm);

    echo "Found " . count($results) . " results\n\n";

    // Process results
    foreach ($results as $idx => $result) {
        echo "Result " . ($idx + 1) . ":\n";
        echo "  Page: " . ($result->pageIndex + 1) . "\n";
        echo "  Text: '" . $result->text . "'\n";
        echo "  Position: " . $result->position . "\n";
        echo "  Bbox: ({$result->boundingBox->x}, {$result->boundingBox->y}, "
             . "{$result->boundingBox->width}, {$result->boundingBox->height})\n\n";
    }

    $pdf->close();

} catch (\Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
    exit(1);
}
