<?php

declare(strict_types=1);

/**
 * Basic PDF Reading Example
 *
 * This example demonstrates how to read a PDF file and extract
 * basic information like page count and text content.
 */

require dirname(__DIR__) . '/vendor/autoload.php';

use PdfOxide\PdfDocument;
use PdfOxide\Exceptions\PdfException;

try {
    // Open a PDF document
    $pdfPath = $argv[1] ?? './sample.pdf';

    if (!file_exists($pdfPath)) {
        echo "Error: File not found: $pdfPath\n";
        echo "Usage: php 01_basic_reading.php <path-to-pdf>\n";
        exit(1);
    }

    echo "Opening PDF: $pdfPath\n";
    $pdf = new PdfDocument($pdfPath);

    // Get document metadata
    echo "\n=== Document Information ===\n";
    $metadata = $pdf->getMetadata();
    echo "File: " . $metadata['file_path'] . "\n";
    echo "Size: " . number_format($metadata['file_size']) . " bytes\n";
    echo "Pages: " . $metadata['page_count'] . "\n";

    $version = $metadata['version'];
    echo "PDF Version: " . $version['major'] . "." . $version['minor'] . "\n";
    echo "Has Structure Tree: " . ($metadata['has_structure_tree'] ? 'Yes' : 'No') . "\n";

    // Extract text from first page
    if ($metadata['page_count'] > 0) {
        echo "\n=== First Page Content ===\n";
        $text = $pdf->extractText(0);
        echo substr($text, 0, 500) . "...\n";
    }

    // List all available information
    echo "\n=== Page-by-Page Overview ===\n";
    for ($page = 0; $page < min(3, $metadata['page_count']); $page++) {
        echo "\nPage " . ($page + 1) . ":\n";

        // Get fonts
        $fonts = $pdf->getFonts($page);
        if (count($fonts) > 0) {
            echo "  Fonts: " . implode(', ', array_map(fn($f) => $f->name, $fonts)) . "\n";
        }

        // Get images
        $images = $pdf->getImages($page);
        echo "  Images: " . count($images) . "\n";

        // Get annotations
        $annotations = $pdf->getAnnotations($page);
        echo "  Annotations: " . count($annotations) . "\n";
    }

    $pdf->close();
    echo "\n✅ Done!\n";

} catch (PdfException $e) {
    echo "PDF Error: " . $e->getMessage() . "\n";
    echo "Code: " . $e->getErrorCode() . "\n";
    exit(1);
} catch (\Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
    exit(1);
}
