<?php

declare(strict_types=1);

namespace PdfOxide\FFI;

use FFI;
use RuntimeException;

/**
 * Loads and manages the native pdf_oxide library via FFI.
 *
 * Handles platform-specific library loading and caching.
 */
class NativeLibrary
{
    /**
     * The cached FFI instance.
     */
    private static ?FFI $ffi = null;

    /**
     * Whether the library has been initialized.
     */
    private static bool $initialized = false;

    /**
     * Get or create the FFI instance.
     *
     * @throws RuntimeException if library cannot be loaded
     */
    public static function getInstance(): FFI
    {
        if (self::$ffi !== null) {
            return self::$ffi;
        }

        self::$ffi = self::loadLibrary();
        self::$initialized = true;

        // Register cleanup on shutdown
        register_shutdown_function([self::class, 'cleanup']);

        return self::$ffi;
    }

    /**
     * Shutdown hook: drop the cached FFI instance so PHP's GC can
     * unmap the dlopen'd cdylib before request teardown. Idempotent;
     * safe to call from `register_shutdown_function`.
     */
    public static function cleanup(): void
    {
        self::$ffi = null;
        self::$initialized = false;
    }

    /**
     * Load the native PDF Oxide library.
     *
     * @throws RuntimeException if library cannot be loaded
     */
    private static function loadLibrary(): FFI
    {
        // Check if FFI extension is available
        if (!extension_loaded('ffi')) {
            throw new RuntimeException(
                'FFI extension is not loaded. Install it with: php -m | grep ffi'
            );
        }

        $headerPath = self::getHeaderPath();
        if (!file_exists($headerPath)) {
            throw new RuntimeException("C header not found at: {$headerPath}");
        }

        $libraryPath = self::findLibrary();
        if (!file_exists($libraryPath)) {
            throw new RuntimeException("PDF Oxide library not found. Searched: {$libraryPath}");
        }

        try {
            $header = file_get_contents($headerPath);
            return FFI::cdef($header, $libraryPath);
        } catch (\FFI\Exception $e) {
            throw new RuntimeException(
                "Failed to load PDF Oxide library: " . $e->getMessage(),
                0,
                $e
            );
        }
    }

    /**
     * Get the C header file path.
     */
    private static function getHeaderPath(): string
    {
        // Try multiple locations
        $candidates = [
            // Relative to this file (in-tree dev / composer-installed)
            __DIR__ . '/../../include/pdf_oxide.h',
            // Composer vendor directory — package name `oxide/pdf-oxide`
            dirname(__DIR__, 3) . '/vendor/oxide/pdf-oxide/include/pdf_oxide.h',
            // Installed in /usr/include
            '/usr/include/pdf_oxide.h',
            '/usr/local/include/pdf_oxide.h',
        ];

        foreach ($candidates as $path) {
            $real = realpath($path);
            if ($real !== false && file_exists($real)) {
                return $real;
            }
        }

        // Return the primary location (for error reporting)
        return $candidates[0];
    }

    /**
     * Find the native library file for the current platform.
     *
     * @throws RuntimeException if library cannot be found
     */
    private static function findLibrary(): string
    {
        $platform = self::detectPlatform();
        $libName = self::getLibraryName($platform);

        // Search paths in order of preference
        $searchPaths = self::getSearchPaths($platform);

        foreach ($searchPaths as $basePath) {
            $fullPath = rtrim($basePath, '/') . '/' . $libName;
            if (file_exists($fullPath) && is_readable($fullPath)) {
                return realpath($fullPath);
            }
        }

        // Try the platform-specific compiled path
        $compiledPath = dirname(__DIR__, 3) . '/target/release/' . $libName;
        if (file_exists($compiledPath) && is_readable($compiledPath)) {
            return realpath($compiledPath);
        }

        throw new RuntimeException(
            sprintf(
                'PDF Oxide library not found for %s. Searched paths: %s',
                $platform,
                implode(', ', $searchPaths)
            )
        );
    }

    /**
     * Detect the current platform.
     *
     * @return string One of: 'linux', 'macos', 'windows'
     */
    private static function detectPlatform(): string
    {
        $os = php_uname('s');

        if (stripos($os, 'Linux') !== false) {
            return 'linux';
        }
        if (stripos($os, 'Darwin') !== false) {
            return 'macos';
        }
        if (stripos($os, 'Windows') !== false) {
            return 'windows';
        }

        throw new RuntimeException("Unsupported platform: {$os}");
    }

    /**
     * Platform key matching the layout used by
     * `scripts/download-native-lib.php` (e.g. `linux-x86_64`).
     * Falls back to the coarse platform name when the architecture is
     * unknown, which leaves the generic <package-root>/lib search path
     * as the next-best candidate.
     */
    private static function detectPlatformKey(string $platform): string
    {
        $arch = strtolower(php_uname('m'));
        $normArch = match (true) {
            $arch === 'x86_64' || $arch === 'amd64' => 'x86_64',
            $arch === 'aarch64' || $arch === 'arm64' => 'aarch64',
            default => $arch,
        };
        return match ($platform) {
            'linux' => 'linux-' . $normArch,
            'macos' => 'darwin-' . ($normArch === 'aarch64' ? 'arm64' : $normArch),
            'windows' => 'windows-x64',
            default => $platform,
        };
    }

    /**
     * Get the library filename for a platform.
     */
    private static function getLibraryName(string $platform): string
    {
        return match ($platform) {
            'linux' => 'libpdf_oxide.so',
            'macos' => 'libpdf_oxide.dylib',
            'windows' => 'pdf_oxide.dll',
            default => throw new RuntimeException("Unknown platform: {$platform}"),
        };
    }

    /**
     * Get library search paths for a platform.
     *
     * @return string[] Array of paths to search
     */
    private static function getSearchPaths(string $platform): array
    {
        $packageRoot = dirname(__DIR__, 3);
        $platformKey = self::detectPlatformKey($platform);
        $paths = [
            // Composer post-install staging:
            //   <package-root>/lib/<platform-key>/<libname>
            // (see scripts/download-native-lib.php).
            $packageRoot . '/lib/' . $platformKey,
            // Generic project-root fallbacks.
            $packageRoot . '/lib',
            $packageRoot . '/bin',
        ];

        // Platform-specific paths
        switch ($platform) {
            case 'linux':
                $paths = array_merge($paths, [
                    '/usr/lib',
                    '/usr/local/lib',
                    '/usr/lib/x86_64-linux-gnu',
                    '/usr/lib64',
                ]);
                // Add LD_LIBRARY_PATH
                if ($ldPath = getenv('LD_LIBRARY_PATH')) {
                    $paths = array_merge($paths, explode(':', $ldPath));
                }
                break;

            case 'macos':
                $paths = array_merge($paths, [
                    '/usr/local/lib',
                    '/opt/homebrew/lib',
                    '/usr/lib',
                ]);
                // Add DYLD_LIBRARY_PATH
                if ($dyldPath = getenv('DYLD_LIBRARY_PATH')) {
                    $paths = array_merge($paths, explode(':', $dyldPath));
                }
                break;

            case 'windows':
                $paths = array_merge($paths, [
                    'C:\\Program Files\\pdf_oxide',
                    'C:\\Program Files (x86)\\pdf_oxide',
                ]);
                // Add PATH
                if ($path = getenv('PATH')) {
                    $paths = array_merge($paths, explode(';', $path));
                }
                break;
        }

        return array_filter($paths, fn($p) => !empty($p));
    }

}
