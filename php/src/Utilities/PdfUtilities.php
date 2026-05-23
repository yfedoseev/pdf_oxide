<?php

declare(strict_types=1);

namespace PdfOxide\Utilities;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Utility functions for common PDF operations.
 *
 * Provides helper methods for encoding, platform detection, and performance tuning.
 */
class PdfUtilities
{
    private static FunctionBindings $bindings;

    private static function ensureBindings(): void
    {
        if (!isset(self::$bindings)) {
            self::$bindings = new FunctionBindings();
        }
    }

    /**
     * Get platform information.
     */
    public static function getPlatformInfo(): array
    {
        self::ensureBindings();
        $info = self::$bindings->pdfGetPlatformInfo();
        return [
            'platform' => $info,
            'is_64bit' => PHP_INT_SIZE === 8,
            'php_version' => PHP_VERSION,
            'os' => PHP_OS_FAMILY,
        ];
    }

    /**
     * Get library version.
     */
    public static function getLibraryVersion(): string
    {
        self::ensureBindings();
        return self::$bindings->pdfGetLibraryVersion();
    }

    /**
     * Check if a feature is supported.
     */
    public static function isFeatureSupported(string $feature): bool
    {
        self::ensureBindings();
        return self::$bindings->pdfIsFeatureSupported($feature);
    }

    /**
     * Get system CPU count.
     */
    public static function getCpuCount(): int
    {
        self::ensureBindings();
        return self::$bindings->pdfGetCpuCount();
    }

    /**
     * Get available memory.
     */
    public static function getAvailableMemory(): int
    {
        self::ensureBindings();
        return self::$bindings->pdfGetAvailableMemory();
    }

    /**
     * Encode data as hex string.
     */
    public static function encodeHex(string $data): string
    {
        self::ensureBindings();
        return self::$bindings->pdfEncodeHex($data);
    }

    /**
     * Decode hex string to data.
     */
    public static function decodeHex(string $hex): string
    {
        self::ensureBindings();
        return self::$bindings->pdfDecodeHex($hex);
    }

    /**
     * Encode data as base64.
     */
    public static function encodeBase64(string $data): string
    {
        self::ensureBindings();
        return self::$bindings->pdfEncodeBase64($data);
    }

    /**
     * Decode base64 data.
     */
    public static function decodeBase64(string $base64): string
    {
        self::ensureBindings();
        return self::$bindings->pdfDecodeBase64($base64);
    }

    /**
     * Get supported image formats.
     */
    public static function getSupportedImageFormats(): array
    {
        self::ensureBindings();
        return self::$bindings->pdfGetSupportedImageFormats();
    }

    /**
     * Enable or disable caching.
     */
    public static function setCachingEnabled(bool $enabled): void
    {
        self::ensureBindings();
        self::$bindings->pdfSetCachingEnabled($enabled);
    }

    /**
     * Set thread pool size for parallel processing.
     */
    public static function setThreadPoolSize(int $size): void
    {
        self::ensureBindings();
        self::$bindings->pdfSetThreadPoolSize($size);
    }

    /**
     * Get performance metrics.
     */
    public static function getPerformanceMetrics(): array
    {
        self::ensureBindings();
        return self::$bindings->pdfGetPerformanceMetrics();
    }

    /**
     * Reset performance metrics.
     */
    public static function resetPerformanceMetrics(): void
    {
        self::ensureBindings();
        self::$bindings->pdfResetPerformanceMetrics();
    }

    /**
     * Detect if data is likely a PDF.
     */
    public static function isPdfData(string $data): bool
    {
        self::ensureBindings();
        return self::$bindings->pdfIsPdfData($data);
    }

    /**
     * Get system capabilities summary.
     */
    public static function getSystemCapabilities(): array
    {
        $capabilities = [];
        self::ensureBindings();

        // Collect all capabilities
        $capabilities['platform'] = self::getPlatformInfo();
        $capabilities['library_version'] = self::getLibraryVersion();
        $capabilities['cpu_count'] = self::getCpuCount();
        $capabilities['available_memory'] = self::getAvailableMemory();
        $capabilities['supported_formats'] = self::getSupportedImageFormats();

        // Check for specific features
        $features = ['ocr', 'signatures', 'compliance', 'rendering', 'barcodes'];
        $capabilities['features'] = [];
        foreach ($features as $feature) {
            try {
                $capabilities['features'][$feature] = self::isFeatureSupported($feature);
            } catch (\Exception $e) {
                $capabilities['features'][$feature] = false;
            }
        }

        return $capabilities;
    }

    /**
     * Format bytes as human-readable string.
     */
    public static function formatBytes(int $bytes, int $precision = 2): string
    {
        $units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
        $bytes = max($bytes, 0);
        $pow = floor(($bytes ? log($bytes) : 0) / log(1024));
        $pow = min($pow, count($units) - 1);
        $bytes /= (1 << (10 * $pow));

        return round($bytes, $precision) . ' ' . $units[$pow];
    }

    /**
     * Get system information summary for logging/debugging.
     */
    public static function getSystemInfo(): string
    {
        $info = self::getSystemCapabilities();
        return sprintf(
            "Platform: %s | Lib: %s | CPUs: %d | Memory: %s | Features: %s",
            $info['platform']['platform'] ?? 'unknown',
            $info['library_version'] ?? 'unknown',
            $info['cpu_count'],
            self::formatBytes($info['available_memory']),
            implode(',', array_keys(array_filter($info['features'])))
        );
    }
}
