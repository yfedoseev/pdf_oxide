<?php

/**
 * 01 — Extract text (PHP)
 *
 * Opens a PDF, prints the page count, then the text of each page.
 *
 *   php main.php ../../../tests/fixtures/simple.pdf
 *
 * Requires `composer install` in the repo root (or php/) so the PdfOxide
 * autoloader is available, and the cdylib discoverable (PDF_OXIDE_CDYLIB_DIR).
 */

declare(strict_types=1);

use PdfOxide\PdfDocument;

// Locate the Composer autoloader (repo-root vendor/ or php/vendor/).
$autoloads = [
    __DIR__ . '/../../../vendor/autoload.php',
    __DIR__ . '/../../../php/vendor/autoload.php',
];
$loaded = false;
foreach ($autoloads as $a) {
    if (is_file($a)) {
        require $a;
        $loaded = true;
        break;
    }
}
if (!$loaded) {
    fwrite(STDERR, "Composer autoloader not found; run `composer install`.\n");
    exit(1);
}

$path = $argv[1] ?? null;
if ($path === null) {
    fwrite(STDERR, "usage: php main.php <pdf>\n");
    exit(1);
}

$doc = PdfDocument::open($path);
$pages = $doc->pageCount();
echo "Pages: {$pages}\n";
for ($i = 0; $i < $pages; $i++) {
    echo '--- Page ' . ($i + 1) . " ---\n";
    echo $doc->extractText($i) . "\n";
}
