<?php

declare(strict_types=1);

/**
 * PDF Creation Example
 *
 * This example demonstrates how to create a new PDF document
 * with text, images, shapes, and custom formatting.
 */

require dirname(__DIR__) . '/vendor/autoload.php';

use PdfOxide\Pdf;
use PdfOxide\Types\Color;
use PdfOxide\Enums\PageSize;

try {
    echo "Creating new PDF document...\n";

    // Create new PDF
    $pdf = Pdf::create();

    // Add a page with A4 size
    $pdf->addPageWithSize(PageSize::A4);

    // Add header
    $pdf->setFont('Helvetica-Bold', 20)
        ->setColor(Color::blue())
        ->text('Sample PDF Document', 50, 50);

    // Add separator line
    $pdf->setColor(Color::gray())
        ->line(50, 80, 545, 80, 2);

    // Add some text with different sizes
    $pdf->setFont('Helvetica', 12)
        ->setColor(Color::black())
        ->text('This is a sample PDF created with PDF Oxide PHP binding.', 50, 100);

    $pdf->setFont('Helvetica', 10)
        ->setColor(Color::fromHex('#666666'))
        ->text('Demonstration of PDF creation capabilities', 50, 130);

    // Add content sections
    $y = 180;

    $pdf->setFont('Helvetica-Bold', 14)
        ->setColor(Color::blue())
        ->text('Features', 50, $y);

    $y += 30;

    $features = [
        '✓ Text rendering with custom fonts and sizes',
        '✓ Color support (RGB with alpha)',
        '✓ Shape drawing (lines, rectangles, circles)',
        '✓ Multiple pages in single document',
        '✓ Image insertion (JPG, PNG, WebP)',
        '✓ Fluent API for easy usage',
    ];

    $pdf->setFont('Helvetica', 11)
        ->setColor(Color::black());

    foreach ($features as $feature) {
        $pdf->text($feature, 70, $y);
        $y += 25;
    }

    // Draw a decorative shape
    $pdf->setColor(Color::fromRgbFloat(0.9, 0.9, 0.9))
        ->rect(50, $y + 10, 495, 100, true, 0);

    $pdf->setFont('Helvetica-Bold', 11)
        ->setColor(Color::fromHex('#333333'))
        ->text('Sample Information Box', 60, $y + 25);

    $pdf->setFont('Helvetica', 10)
        ->setColor(Color::fromHex('#555555'))
        ->text('This PDF was created programmatically using PDF Oxide.', 60, $y + 50)
        ->text('You can create professional documents with just a few lines of PHP code.', 60, $y + 70);

    // Add footer
    $y = 800;
    $pdf->setFont('Helvetica', 9)
        ->setColor(Color::gray())
        ->text('Generated: ' . date('Y-m-d H:i:s'), 50, $y)
        ->text('PDF Oxide v0.3.3 PHP Binding', 450, $y);

    // Save document
    $outputFile = 'generated_sample.pdf';
    echo "Saving to: $outputFile\n";
    $pdf->save($outputFile);

    echo "\n✅ PDF created successfully!\n";
    echo "File: $outputFile\n";
    echo "Size: " . number_format(filesize($outputFile)) . " bytes\n";

} catch (\Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
    if ($e instanceof \PdfOxide\Exceptions\PdfException) {
        echo "Code: " . $e->getErrorCode() . "\n";
    }
    exit(1);
}
