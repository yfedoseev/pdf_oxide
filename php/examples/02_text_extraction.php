<?php

declare(strict_types=1);

/**
 * Text Extraction and Search Example
 *
 * This example demonstrates how to extract and convert text,
 * as well as perform full-text search operations.
 */

require dirname(__DIR__) . '/vendor/autoload.php';

use PdfOxide\PdfDocument;
use PdfOxide\Exceptions\PdfException;

try {
    $pdfPath = $argv[1] ?? './sample.pdf';

    if (!file_exists($pdfPath)) {
        echo "Error: File not found: $pdfPath\n";
        echo "Usage: php 02_text_extraction.php <path-to-pdf> [search-term]\n";
        exit(1);
    }

    $pdf = new PdfDocument($pdfPath);
    $searchTerm = $argv[2] ?? null;

    // Extract text in different formats from first page
    if ($pdf->getPageCount() > 0) {
        echo "=== First Page - Plain Text ===\n";
        $plainText = $pdf->toPlainText(0);
        echo substr($plainText, 0, 300) . "...\n\n";

        echo "=== First Page - Markdown ===\n";
        $markdown = $pdf->toMarkdown(0);
        echo substr($markdown, 0, 300) . "...\n\n";

        echo "=== First Page - HTML ===\n";
        $html = $pdf->toHtml(0);
        echo substr($html, 0, 300) . "...\n\n";
    }

    // Perform search if term provided
    if ($searchTerm) {
        echo "=== Searching for: \"$searchTerm\" ===\n";

        // Search entire document
        $results = $pdf->searchAll($searchTerm, caseSensitive: false);

        if (count($results) === 0) {
            echo "No results found.\n";
        } else {
            echo "Found " . count($results) . " result(s):\n\n";

            foreach (array_slice($results, 0, 5) as $idx => $result) {
                echo ($idx + 1) . ". Page " . ($result->pageIndex + 1) . " at position " . $result->position . "\n";
                echo "   Text: " . substr($result->text, 0, 100) . "\n";
                echo "   Bbox: [" . $result->boundingBox->x . ", " . $result->boundingBox->y . ", "
                     . $result->boundingBox->width . ", " . $result->boundingBox->height . "]\n\n";
            }

            if (count($results) > 5) {
                echo "... and " . (count($results) - 5) . " more results.\n";
            }
        }

        // Search specific page
        if ($pdf->getPageCount() > 0) {
            echo "\n=== Searching first page only ===\n";
            $pageResults = $pdf->searchPage($searchTerm, 0, caseSensitive: false);
            echo "Found " . count($pageResults) . " result(s) on first page.\n";
        }
    } else {
        echo "Tip: Provide a search term to search the document:\n";
        echo "php 02_text_extraction.php '$pdfPath' 'search-term'\n";
    }

    // Extract all content to markdown file
    if ($pdf->getPageCount() > 0) {
        echo "\n=== Extracting full document to Markdown ===\n";
        $allMarkdown = $pdf->toMarkdownAll();
        $outputFile = 'extracted_' . basename($pdfPath, '.pdf') . '.md';
        file_put_contents($outputFile, $allMarkdown);
        echo "Saved to: $outputFile (" . strlen($allMarkdown) . " bytes)\n";
    }

    $pdf->close();
    echo "\n✅ Done!\n";

} catch (PdfException $e) {
    echo "PDF Error: " . $e->getMessage() . "\n";
    exit(1);
} catch (\Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
    exit(1);
}
