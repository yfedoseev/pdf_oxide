<?php

declare(strict_types=1);

/**
 * PHP-CS-Fixer configuration for pdf_oxide PHP binding.
 *
 * Uses the @PER-CS2.0 preset (PHP-FIG PER Coding Style 2.0), which
 * supersedes PSR-12 and covers match-expression formatting, attribute
 * grouping, and enum styling that PSR-12 predates.
 *
 * Run locally:   composer cs-fix         (in-place)
 * Check in CI:   composer cs-check       (dry-run + diff)
 */

$finder = PhpCsFixer\Finder::create()
    ->in([
        __DIR__ . '/src',
        __DIR__ . '/tests',
    ])
    ->name('*.php')
    ->notPath('vendor')
    ->ignoreDotFiles(true)
    ->ignoreVCS(true);

$config = new PhpCsFixer\Config();

return $config
    ->setRiskyAllowed(true)
    ->setRules([
        '@PER-CS2.0'           => true,
        '@PER-CS2.0:risky'     => true,
        '@PHP82Migration'      => true,
        '@PHP82Migration:risky' => true,
        'declare_strict_types' => true,
        'no_unused_imports'    => true,
        'ordered_imports'      => [
            'sort_algorithm' => 'alpha',
            'imports_order'  => ['class', 'function', 'const'],
        ],
        'trailing_comma_in_multiline' => true,
    ])
    ->setFinder($finder);
