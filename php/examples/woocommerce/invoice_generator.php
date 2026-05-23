<?php

declare(strict_types=1);

/**
 * WooCommerce Invoice Generator
 *
 * Generates professional invoices for WooCommerce orders using PDF Oxide.
 * This example shows how to create a complete invoice with:
 * - Header with company info
 * - Order details
 * - Item table
 * - Calculations (subtotal, tax, total)
 * - Footer with terms
 */

require dirname(__DIR__, 2) . '/vendor/autoload.php';

use PdfOxide\Pdf;
use PdfOxide\Types\Color;
use PdfOxide\Enums\PageSize;

/**
 * Generate an invoice PDF for a WooCommerce order.
 *
 * @param array $orderData Order data array with structure:
 *   [
 *     'order_id' => '12345',
 *     'order_number' => '#12345',
 *     'date' => '2025-01-22',
 *     'status' => 'completed',
 *     'billing' => ['name', 'email', 'phone', 'address', 'city', 'state', 'postcode', 'country'],
 *     'shipping' => ['address', 'city', 'state', 'postcode', 'country'],
 *     'items' => [
 *       ['name' => 'Product Name', 'qty' => 1, 'price' => 29.99, 'total' => 29.99],
 *       ...
 *     ],
 *     'subtotal' => 100.00,
 *     'shipping_total' => 10.00,
 *     'tax_total' => 8.80,
 *     'total' => 118.80,
 *     'currency' => 'USD',
 *   ]
 * @return string The PDF content as string
 */
function generateInvoicePdf(array $orderData): string
{
    $pdf = Pdf::create();

    // Add page with margins
    $pdf->addPageWithSize(PageSize::A4);
    $marginTop = 40;
    $marginLeft = 40;
    $marginRight = 40;
    $pageWidth = 595 - $marginLeft - $marginRight;

    // ===== HEADER SECTION =====
    $y = $marginTop;

    // Company logo/name placeholder
    $pdf->setFont('Helvetica-Bold', 16)
        ->setColor(Color::blue())
        ->text('YOUR COMPANY NAME', $marginLeft, $y);

    $y += 20;

    $pdf->setFont('Helvetica', 9)
        ->setColor(Color::fromHex('#666666'))
        ->text('123 Business Street, City, State 12345', $marginLeft, $y)
        ->text('Email: business@example.com | Phone: (555) 123-4567', $marginLeft, $y + 12);

    // Horizontal line
    $y = 120;
    $pdf->setColor(Color::gray())
        ->line($marginLeft, $y, 595 - $marginRight, $y, 1);

    // ===== INVOICE INFO SECTION =====
    $y = 135;

    $pdf->setFont('Helvetica-Bold', 14)
        ->setColor(Color::black())
        ->text('INVOICE', $marginLeft, $y);

    $y += 25;

    // Invoice details (left side)
    $pdf->setFont('Helvetica-Bold', 10)
        ->text('Invoice Number:', $marginLeft, $y);

    $pdf->setFont('Helvetica', 10)
        ->setColor(Color::black())
        ->text($orderData['order_number'], $marginLeft + 95, $y);

    $y += 15;

    $pdf->setFont('Helvetica-Bold', 10)
        ->text('Invoice Date:', $marginLeft, $y);

    $pdf->setFont('Helvetica', 10)
        ->text(date('F j, Y', strtotime($orderData['date'])), $marginLeft + 95, $y);

    // Order details (right side)
    $rightCol = 350;

    $pdf->setFont('Helvetica-Bold', 10)
        ->text('Order Date:', $rightCol, $y);

    $pdf->setFont('Helvetica', 10)
        ->text(date('F j, Y', strtotime($orderData['date'])), $rightCol + 75, $y);

    $y = 175;

    $pdf->setFont('Helvetica-Bold', 10)
        ->text('Due Date:', $marginLeft, $y);

    $pdf->setFont('Helvetica', 10)
        ->text(date('F j, Y', strtotime('+30 days', strtotime($orderData['date']))), $marginLeft + 95, $y);

    // ===== CUSTOMER INFO SECTION =====
    $y = 220;

    // Billing info
    $pdf->setFont('Helvetica-Bold', 10)
        ->setColor(Color::blue())
        ->text('BILL TO:', $marginLeft, $y);

    $y += 15;

    $pdf->setFont('Helvetica', 10)
        ->setColor(Color::black());

    $billing = $orderData['billing'];
    $billingLines = [
        $billing['name'] ?? '',
        $billing['address'] ?? '',
        ($billing['city'] ?? '') . ', ' . ($billing['state'] ?? '') . ' ' . ($billing['postcode'] ?? ''),
        $billing['country'] ?? '',
    ];

    foreach (array_filter($billingLines) as $line) {
        $pdf->text($line, $marginLeft, $y);
        $y += 12;
    }

    // Shipping info
    $y = 220;

    $pdf->setFont('Helvetica-Bold', 10)
        ->setColor(Color::blue())
        ->text('SHIP TO:', $rightCol, $y);

    $y += 15;

    $pdf->setFont('Helvetica', 10)
        ->setColor(Color::black());

    $shipping = $orderData['shipping'] ?? $orderData['billing'];
    $shippingLines = [
        $shipping['address'] ?? '',
        ($shipping['city'] ?? '') . ', ' . ($shipping['state'] ?? '') . ' ' . ($shipping['postcode'] ?? ''),
        $shipping['country'] ?? '',
    ];

    foreach (array_filter($shippingLines) as $line) {
        $pdf->text($line, $rightCol, $y);
        $y += 12;
    }

    // ===== ITEMS TABLE =====
    $y = 330;

    // Table header
    $pdf->setColor(Color::fromHex('#E8E8E8'))
        ->rect($marginLeft, $y, $pageWidth, 20, true, 0);

    $pdf->setFont('Helvetica-Bold', 10)
        ->setColor(Color::black())
        ->text('Description', $marginLeft + 5, $y + 5)
        ->text('Qty', 420, $y + 5)
        ->text('Price', 460, $y + 5)
        ->text('Total', 520, $y + 5);

    $y += 25;

    // Table rows
    $pdf->setFont('Helvetica', 9);
    $rowHeight = 18;

    foreach ($orderData['items'] as $item) {
        $itemName = $item['name'];
        $itemQty = (int)$item['qty'];
        $itemPrice = number_format($item['price'], 2);
        $itemTotal = number_format($item['total'], 2);

        $pdf->text($itemName, $marginLeft + 5, $y)
            ->text($itemQty, 425, $y)
            ->text('$' . $itemPrice, 460, $y)
            ->text('$' . $itemTotal, 520, $y);

        $y += $rowHeight;
    }

    // ===== TOTALS SECTION =====
    $y += 10;

    $totalsX = 450;

    // Subtotal
    $pdf->setFont('Helvetica', 10)
        ->text('Subtotal:', $totalsX, $y)
        ->text('$' . number_format($orderData['subtotal'], 2), 525, $y);

    $y += 16;

    // Shipping
    $pdf->text('Shipping:', $totalsX, $y)
        ->text('$' . number_format($orderData['shipping_total'], 2), 525, $y);

    $y += 16;

    // Tax
    $pdf->text('Tax:', $totalsX, $y)
        ->text('$' . number_format($orderData['tax_total'], 2), 525, $y);

    $y += 20;

    // Total (highlighted)
    $pdf->setColor(Color::fromHex('#F0F0F0'))
        ->rect(440, $y - 5, 95, 20, true, 0);

    $pdf->setFont('Helvetica-Bold', 12)
        ->setColor(Color::blue())
        ->text('TOTAL:', $totalsX, $y)
        ->text('$' . number_format($orderData['total'], 2), 525, $y);

    // ===== FOOTER =====
    $y = 750;

    $pdf->setColor(Color::gray())
        ->line($marginLeft, $y, 595 - $marginRight, $y, 1);

    $y += 15;

    $pdf->setFont('Helvetica', 8)
        ->setColor(Color::fromHex('#999999'))
        ->text('Thank you for your business!', $marginLeft, $y)
        ->text('Payment Terms: Net 30 | Please include invoice number with payment', $marginLeft, $y + 10);

    return $pdf->saveToString();
}

// ===== EXAMPLE USAGE =====

// Sample order data
$sampleOrder = [
    'order_id' => '12345',
    'order_number' => '#12345',
    'date' => '2025-01-22',
    'status' => 'completed',
    'billing' => [
        'name' => 'John Doe',
        'email' => 'john@example.com',
        'phone' => '(555) 123-4567',
        'address' => '123 Main Street',
        'city' => 'New York',
        'state' => 'NY',
        'postcode' => '10001',
        'country' => 'USA',
    ],
    'shipping' => [
        'address' => '456 Oak Avenue',
        'city' => 'Boston',
        'state' => 'MA',
        'postcode' => '02101',
        'country' => 'USA',
    ],
    'items' => [
        [
            'name' => 'Premium Widget Pack',
            'qty' => 2,
            'price' => 29.99,
            'total' => 59.98,
        ],
        [
            'name' => 'Extended Warranty',
            'qty' => 1,
            'price' => 19.99,
            'total' => 19.99,
        ],
        [
            'name' => 'Installation Service',
            'qty' => 1,
            'price' => 20.00,
            'total' => 20.00,
        ],
    ],
    'subtotal' => 99.97,
    'shipping_total' => 15.00,
    'tax_total' => 8.77,
    'total' => 123.74,
    'currency' => 'USD',
];

try {
    echo "Generating WooCommerce invoice...\n";
    $pdfContent = generateInvoicePdf($sampleOrder);

    $outputFile = 'invoice_' . $sampleOrder['order_number'] . '.pdf';
    file_put_contents($outputFile, $pdfContent);

    echo "\n✅ Invoice generated successfully!\n";
    echo "File: $outputFile\n";
    echo "Size: " . number_format(filesize($outputFile)) . " bytes\n";

} catch (\Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
    if ($e instanceof \PdfOxide\Exceptions\PdfException) {
        echo "Code: " . $e->getErrorCode() . "\n";
    }
    exit(1);
}
