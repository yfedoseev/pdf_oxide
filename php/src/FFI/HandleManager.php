<?php

declare(strict_types=1);

namespace PdfOxide\FFI;

use FFI\CData;
use WeakMap;

/**
 * Manages handle lifecycle and cleanup.
 *
 * Tracks all open handles to prevent memory leaks.
 */
class HandleManager
{
    /**
     * Active handles with their types and creation timestamps.
     * @var array<string, array{type: string, created: float, resource: CData}>
     */
    private static array $activeHandles = [];

    /**
     * Handle counter for debugging.
     */
    private static int $handleCounter = 0;

    /**
     * Shutdown handler registered flag.
     */
    private static bool $shutdownHandlerRegistered = false;

    /**
     * Register a new handle.
     *
     * @param CData $handle The handle to track
     * @param string $type The handle type (e.g., 'PdfDocumentHandle')
     * @param string $identifier Optional identifier for debugging
     */
    public static function register(CData $handle, string $type, string $identifier = ''): void
    {
        $id = self::getHandleId($handle);

        self::$activeHandles[$id] = [
            'type' => $type,
            'created' => microtime(true),
            'resource' => $handle,
            'identifier' => $identifier,
        ];

        self::$handleCounter++;

        // Register shutdown handler once
        if (!self::$shutdownHandlerRegistered) {
            register_shutdown_function([self::class, 'cleanup']);
            self::$shutdownHandlerRegistered = true;
        }
    }

    /**
     * Unregister a handle when it's freed.
     *
     * @param CData $handle The handle to unregister
     */
    public static function unregister(CData $handle): void
    {
        $id = self::getHandleId($handle);
        unset(self::$activeHandles[$id]);
    }

    /**
     * Get a unique identifier for a handle.
     */
    private static function getHandleId(CData $handle): string
    {
        return spl_object_hash((object)$handle);
    }

    /**
     * Get all active handles.
     *
     * @return array<string, array> Array of active handles
     */
    public static function getActive(): array
    {
        return self::$activeHandles;
    }

    /**
     * Get handle count.
     *
     * @return int Number of active handles
     */
    public static function count(): int
    {
        return count(self::$activeHandles);
    }

    /**
     * Check if a handle is registered.
     *
     * @param CData $handle The handle to check
     * @return bool True if handle is registered
     */
    public static function isRegistered(CData $handle): bool
    {
        $id = self::getHandleId($handle);
        return isset(self::$activeHandles[$id]);
    }

    /**
     * Get information about a registered handle.
     *
     * @param CData $handle The handle
     * @return array|null Handle information or null if not found
     */
    public static function getInfo(CData $handle): ?array
    {
        $id = self::getHandleId($handle);
        return self::$activeHandles[$id] ?? null;
    }

    /**
     * Get statistics about handle usage.
     *
     * @return array Statistics
     */
    public static function getStatistics(): array
    {
        $typeStats = [];
        $totalAge = 0;
        $count = 0;

        foreach (self::$activeHandles as $info) {
            $type = $info['type'];
            if (!isset($typeStats[$type])) {
                $typeStats[$type] = 0;
            }
            $typeStats[$type]++;

            $age = microtime(true) - $info['created'];
            $totalAge += $age;
            $count++;
        }

        return [
            'total_active' => $count,
            'total_created' => self::$handleCounter,
            'by_type' => $typeStats,
            'average_age_seconds' => $count > 0 ? $totalAge / $count : 0,
        ];
    }

    /**
     * Clean up all resources on shutdown.
     *
     * This is called automatically on script shutdown.
     */
    public static function cleanup(): void
    {
        $bindings = new FunctionBindings();

        foreach (self::$activeHandles as $id => $info) {
            try {
                $handle = $info['resource'];
                $type = $info['type'];

                // Free based on type
                match ($type) {
                    'PdfDocumentHandle' => $bindings->pdfDocumentFree($handle),
                    'SearchResultsHandle' => $bindings->oxideSearchResultFree($handle),
                    'AnnotationListHandle' => $bindings->oxideAnnotationFree($handle),
                    'FontListHandle' => $bindings->oxideFontFree($handle),
                    'ImageListHandle' => $bindings->oxideImageFree($handle),
                    default => null, // Unknown type
                };
            } catch (\Exception $e) {
                trigger_error(
                    sprintf(
                        'Error cleaning up %s handle: %s',
                        $info['type'] ?? 'unknown',
                        $e->getMessage()
                    ),
                    E_USER_WARNING
                );
            }
        }

        self::$activeHandles = [];
    }

    /**
     * Reset all tracking (for testing).
     *
     * @internal
     */
    public static function reset(): void
    {
        self::$activeHandles = [];
        self::$handleCounter = 0;
        self::$shutdownHandlerRegistered = false;
    }
}
