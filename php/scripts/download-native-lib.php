<?php

declare(strict_types=1);

/**
 * Composer post-install / post-update hook: fetch the platform-appropriate
 * libpdf_oxide native library from the corresponding GitHub Release.
 *
 * NOTE: stub. Full implementation arrives in Phase 6 of the v0.3.55 PHP
 * workstream. For now this script:
 *   - exits 0 quietly when not invoked from Composer (e.g. CI lint),
 *   - prints a clear advisory if it is invoked, so users know they must
 *     install libpdf_oxide manually until Phase 6 lands.
 *
 * See `docs/releases/plans/v0.3.55/feature-php-binding.md` Phase 7.2 for
 * the planned download semantics:
 *   - resolve {os, arch} via `PHP_OS_FAMILY` + `php_uname('m')`
 *   - download from
 *     https://github.com/fyi-oxide/pdf_oxide/releases/download/v<VER>/
 *     libpdf_oxide-<os>-<arch>.<ext>
 *   - verify SHA-256 against the .sha256 sibling file
 *   - place under vendor/oxide/pdf-oxide/lib/<arch>/<basename>
 *   - NativeLibrary::findLibrary() already searches `<project>/lib/`.
 */

// TODO: Phase 6 implementation — see top-of-file plan citation.
fwrite(STDERR, "[pdf_oxide] post-install hook is a stub in v0.3.55 Phase 5.\n");
fwrite(STDERR, "[pdf_oxide] Install libpdf_oxide manually for now:\n");
fwrite(STDERR, "[pdf_oxide]   Linux:   apt|build → /usr/local/lib/libpdf_oxide.so\n");
fwrite(STDERR, "[pdf_oxide]   macOS:   build    → /usr/local/lib/libpdf_oxide.dylib\n");
fwrite(STDERR, "[pdf_oxide]   Windows: build    → C:\\\\Program Files\\\\pdf_oxide\\\\pdf_oxide.dll\n");
fwrite(STDERR, "[pdf_oxide] Phase 6 will automate this download from GitHub Releases.\n");

exit(0);
