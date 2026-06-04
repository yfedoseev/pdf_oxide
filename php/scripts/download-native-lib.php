<?php

declare(strict_types=1);

/**
 * Composer post-install / post-update hook: download the platform-
 * appropriate `libpdf_oxide` native library from the corresponding
 * GitHub Release.
 *
 * Distribution model mirrors Python (where `pip install pdf_oxide`
 * pulls a wheel bundling the cdylib): a pure-PHP Composer package
 * wraps the same `libpdf_oxide` cdylib used by every other binding;
 * this hook downloads the platform binary on install.
 *
 * Lookup logic:
 *   - Detect (os, arch) via `php_uname('s')` + `php_uname('m')`.
 *   - Map to one of `linux-x86_64`, `linux-aarch64`, `darwin-x86_64`,
 *     `darwin-arm64`, `windows-x64`.
 *   - Compose the GitHub Releases URL for the package's version.
 *   - Stream the archive to a temp file (`curl` shell-out preferred,
 *     `file_get_contents` fallback for environments without curl).
 *   - SHA256-verify against the manifest sibling-shipped with the
 *     package (`scripts/native-manifest.json`).
 *   - Extract `libpdf_oxide.{so,dylib,dll}` into
 *     `<package-root>/lib/<platform>/`.
 *   - `NativeLibrary::findLibrary()` already searches
 *     `<package-root>/lib/`.
 *
 * Failure mode: print a clear advisory + manual install instructions
 * to stderr and exit 1. Per
 * `feedback_extraction_graceful_fallback`, install-time download
 * failure is acceptable to fail loud (this isn't extraction runtime).
 *
 * Per-package overrides:
 *   - Set `PDF_OXIDE_SKIP_DOWNLOAD=1` to skip the post-install download
 *     entirely (CI / offline / corp-proxy use case).
 *   - Set `PDF_OXIDE_NATIVE_VERSION=vX.Y.Z` to pin a specific release.
 */

const PACKAGE_VERSION_DEFAULT = 'v0.3.60';
const RELEASE_BASE_URL = 'https://github.com/yfedoseev/pdf_oxide/releases/download';
// Path is relative to the package root (parent-of-php in the new
// root-composer.json layout); see comment on $packageRoot below.
const MANIFEST_RELATIVE = 'php/scripts/native-manifest.json';

/**
 * Top-level entry point.
 */
function main(): int
{
    if (getenv('PDF_OXIDE_SKIP_DOWNLOAD') === '1') {
        fwrite(STDOUT, "[pdf_oxide] PDF_OXIDE_SKIP_DOWNLOAD=1 — skipping native-lib download.\n");
        return 0;
    }

    // Two-levels-up so the lib lands at `<package-root>/lib/<platform>/`
    // where `<package-root>` is the COMPOSER package root (either
    // `vendor/oxide/pdf-oxide/` for end-user installs, or the repo
    // root for local dev). One level up would put the lib at
    // `<package-root>/php/lib/…`, but `NativeLibrary::getSearchPaths()`
    // resolves the search root via `dirname(__DIR__, 3)` from
    // `php/src/FFI/NativeLibrary.php` which already lands at
    // `<package-root>`, not `<package-root>/php`. Keep these two
    // path computations in sync — they describe the same directory
    // tree from opposite ends.
    $packageRoot = dirname(__DIR__, 2);
    $version = getenv('PDF_OXIDE_NATIVE_VERSION') ?: PACKAGE_VERSION_DEFAULT;

    try {
        $platform = detectPlatform();
    } catch (RuntimeException $e) {
        fwrite(STDERR, "[pdf_oxide] Platform detection failed: " . $e->getMessage() . "\n");
        printManualInstall($version);
        return 1;
    }

    fwrite(STDOUT, "[pdf_oxide] Detected platform: {$platform['key']} (lib: {$platform['lib_name']})\n");

    $libDir = $packageRoot . '/lib/' . $platform['key'];
    $libPath = $libDir . '/' . $platform['lib_name'];

    if (is_file($libPath)) {
        fwrite(STDOUT, "[pdf_oxide] Native library already present at {$libPath} — nothing to do.\n");
        return 0;
    }

    // Try to load the SHA256 manifest if it exists; absence is non-fatal
    // (we still attempt the download — the verify step is best-effort).
    $manifest = loadManifest($packageRoot, $version);

    $archive = sprintf(
        '%s/%s/libpdf_oxide-%s-%s.tar.gz',
        RELEASE_BASE_URL,
        $version,
        $version,
        $platform['key']
    );

    fwrite(STDOUT, "[pdf_oxide] Downloading: {$archive}\n");

    $tmpFile = tempnam(sys_get_temp_dir(), 'pdf_oxide_');
    if ($tmpFile === false) {
        fwrite(STDERR, "[pdf_oxide] Could not create temp file.\n");
        return 1;
    }

    try {
        if (! downloadFile($archive, $tmpFile)) {
            fwrite(STDERR, "[pdf_oxide] Download failed.\n");
            printManualInstall($version);
            return 1;
        }

        if ($manifest !== null && isset($manifest[$platform['key']])) {
            $expectedSha = $manifest[$platform['key']]['sha256'] ?? null;
            if ($expectedSha !== null && ! verifySha256($tmpFile, $expectedSha)) {
                fwrite(STDERR, "[pdf_oxide] SHA256 mismatch — refusing to install.\n");
                unlink($tmpFile);
                return 1;
            }
        } else {
            fwrite(STDOUT, "[pdf_oxide] (no manifest entry for {$platform['key']}; skipping SHA256 verification)\n");
        }

        if (! is_dir($libDir) && ! mkdir($libDir, 0755, true) && ! is_dir($libDir)) {
            fwrite(STDERR, "[pdf_oxide] Could not create {$libDir}\n");
            return 1;
        }

        if (! extractArchive($tmpFile, $libDir, $platform['lib_name'])) {
            fwrite(STDERR, "[pdf_oxide] Archive extraction failed.\n");
            printManualInstall($version);
            return 1;
        }

        if (! is_file($libPath)) {
            fwrite(STDERR, "[pdf_oxide] Native library not present after extraction: {$libPath}\n");
            return 1;
        }

        fwrite(STDOUT, "[pdf_oxide] Installed native library to {$libPath}\n");
        return 0;
    } finally {
        if (is_file($tmpFile)) {
            @unlink($tmpFile);
        }
    }
}

/**
 * Detect the current platform.
 *
 * @return array{key:string, lib_name:string}
 * @throws RuntimeException for unsupported (os, arch) combos.
 */
function detectPlatform(): array
{
    $os = strtolower(php_uname('s'));
    $arch = strtolower(php_uname('m'));

    // Normalize architecture aliases.
    $normArch = match (true) {
        $arch === 'x86_64' || $arch === 'amd64' => 'x86_64',
        $arch === 'aarch64' || $arch === 'arm64' => 'aarch64',
        default => $arch,
    };

    if (str_contains($os, 'linux')) {
        $key = match ($normArch) {
            'x86_64' => 'linux-x86_64',
            'aarch64' => 'linux-aarch64',
            default => throw new RuntimeException("Unsupported Linux arch: {$arch}"),
        };
        return ['key' => $key, 'lib_name' => 'libpdf_oxide.so'];
    }

    if (str_contains($os, 'darwin')) {
        $key = match ($normArch) {
            'x86_64' => 'darwin-x86_64',
            'aarch64' => 'darwin-arm64',
            default => throw new RuntimeException("Unsupported macOS arch: {$arch}"),
        };
        return ['key' => $key, 'lib_name' => 'libpdf_oxide.dylib'];
    }

    if (str_contains($os, 'windows') || str_contains($os, 'win')) {
        if ($normArch !== 'x86_64') {
            throw new RuntimeException("Unsupported Windows arch: {$arch} (only x64 is shipped)");
        }
        return ['key' => 'windows-x64', 'lib_name' => 'pdf_oxide.dll'];
    }

    throw new RuntimeException("Unsupported OS: {$os}");
}

/**
 * Load the optional SHA256 manifest. Format:
 *   { "linux-x86_64": { "sha256": "abc..." }, ... }
 *
 * @return array<string,array<string,string>>|null null if absent.
 */
function loadManifest(string $packageRoot, string $version): ?array
{
    $path = $packageRoot . '/' . MANIFEST_RELATIVE;
    if (! is_file($path)) {
        return null;
    }
    try {
        $raw = file_get_contents($path);
        if ($raw === false) {
            return null;
        }
        $manifest = json_decode($raw, true, 512, JSON_THROW_ON_ERROR);
        if (! is_array($manifest)) {
            return null;
        }
        // Manifests may be top-level platform-keyed, OR keyed by version:
        if (isset($manifest[$version]) && is_array($manifest[$version])) {
            return $manifest[$version];
        }
        return $manifest;
    } catch (\JsonException) {
        return null;
    }
}

/**
 * Download a URL to a local file. Uses `curl` when available for
 * better progress / cert handling; falls back to `file_get_contents`.
 */
function downloadFile(string $url, string $dest): bool
{
    // Try curl first.
    $curlBin = trim((string)@shell_exec('command -v curl'));
    if ($curlBin !== '') {
        $cmd = sprintf(
            '%s -fLso %s %s',
            escapeshellarg($curlBin),
            escapeshellarg($dest),
            escapeshellarg($url)
        );
        $rc = 0;
        $out = [];
        exec($cmd . ' 2>&1', $out, $rc);
        if ($rc === 0 && is_file($dest) && filesize($dest) > 0) {
            return true;
        }
        fwrite(STDERR, "[pdf_oxide] curl failed (rc={$rc}): " . implode("\n", $out) . "\n");
    }

    // Fallback: file_get_contents with TLS context.
    $ctx = stream_context_create([
        'http' => [
            'follow_location' => 1,
            'timeout' => 60,
            'user_agent' => 'pdf_oxide-php-installer/0.3.60',
        ],
        'https' => [
            'follow_location' => 1,
            'timeout' => 60,
            'user_agent' => 'pdf_oxide-php-installer/0.3.60',
        ],
    ]);
    $data = @file_get_contents($url, false, $ctx);
    if ($data === false) {
        return false;
    }
    return file_put_contents($dest, $data) === strlen($data);
}

/**
 * Verify SHA256 of a downloaded file.
 */
function verifySha256(string $path, string $expectedHex): bool
{
    $actual = hash_file('sha256', $path);
    if ($actual === false) {
        return false;
    }
    return hash_equals(strtolower(trim($expectedHex)), $actual);
}

/**
 * Extract a tar.gz archive and place the named library file into the
 * destination directory. Strips any leading directory components in
 * the archive.
 */
function extractArchive(string $archive, string $destDir, string $libName): bool
{
    // Prefer system `tar` (universal on Linux/macOS).
    $tarBin = trim((string)@shell_exec('command -v tar'));
    if ($tarBin !== '') {
        $cmd = sprintf(
            '%s -xzf %s -C %s --strip-components=1 2>&1',
            escapeshellarg($tarBin),
            escapeshellarg($archive),
            escapeshellarg($destDir)
        );
        $rc = 0;
        $out = [];
        exec($cmd, $out, $rc);
        if ($rc === 0) {
            // Validate the expected file landed where we expected.
            return is_file($destDir . '/' . $libName);
        }
        // Some archive layouts don't use a leading directory; retry without strip.
        $cmd = sprintf(
            '%s -xzf %s -C %s 2>&1',
            escapeshellarg($tarBin),
            escapeshellarg($archive),
            escapeshellarg($destDir)
        );
        exec($cmd, $out, $rc);
        if ($rc === 0) {
            return is_file($destDir . '/' . $libName);
        }
    }

    // PharData fallback (works on Windows where `tar` isn't always present).
    try {
        $pharPath = $archive . '.tmp.tar';
        copy($archive, $pharPath);
        $phar = new \PharData($pharPath);
        $phar->decompress();
        $tar = new \PharData(substr($pharPath, 0, -3) . 'tar');
        $tar->extractTo($destDir, null, true);
        @unlink($pharPath);
        return is_file($destDir . '/' . $libName);
    } catch (\Throwable $e) {
        fwrite(STDERR, "[pdf_oxide] PharData extract failed: " . $e->getMessage() . "\n");
        return false;
    }
}

function printManualInstall(string $version): void
{
    fwrite(STDERR, "\n[pdf_oxide] Manual install instructions:\n");
    fwrite(STDERR, "  1. Download libpdf_oxide for your platform:\n");
    fwrite(STDERR, "       https://github.com/yfedoseev/pdf_oxide/releases/tag/{$version}\n");
    fwrite(STDERR, "  2. Extract the cdylib (libpdf_oxide.so / .dylib / pdf_oxide.dll)\n");
    fwrite(STDERR, "     into one of:\n");
    fwrite(STDERR, "       vendor/oxide/pdf-oxide/lib/<platform>/\n");
    fwrite(STDERR, "       /usr/local/lib\n");
    fwrite(STDERR, "       /usr/lib\n");
    fwrite(STDERR, "  3. NativeLibrary::findLibrary() will discover it on next use.\n");
    fwrite(STDERR, "  Or set PDF_OXIDE_SKIP_DOWNLOAD=1 to silence this hook entirely.\n\n");
}

// Only run main() when invoked as the entry script (allows the file to
// be `require`d for testing).
if (PHP_SAPI === 'cli' && realpath($_SERVER['SCRIPT_FILENAME'] ?? '') === __FILE__) {
    exit(main());
}
