<?php

declare(strict_types=1);

/**
 * PHPUnit Bootstrap File
 *
 * Loads the autoloader and sets up test environment.
 */

// Load autoloader
$autoloader = dirname(__DIR__) . '/vendor/autoload.php';
if (!file_exists($autoloader)) {
    throw new RuntimeException('Autoloader not found. Run: composer install');
}

require $autoloader;

// Set error reporting
error_reporting(E_ALL);
ini_set('display_errors', '1');

// Define test fixtures directory
define('TEST_FIXTURES_DIR', __DIR__ . '/Fixtures');

// Ensure fixtures directory exists
if (!is_dir(TEST_FIXTURES_DIR)) {
    mkdir(TEST_FIXTURES_DIR, 0777, true);
}

// Create a simple test PDF if it doesn't exist
if (!file_exists(TEST_FIXTURES_DIR . '/sample.pdf')) {
    // This would ideally be created from a real PDF
    // For now, we'll just note that it needs to be provided
    echo "Note: Test PDF not found at " . TEST_FIXTURES_DIR . "/sample.pdf\n";
    echo "Please add a sample PDF file for integration tests.\n";
}
