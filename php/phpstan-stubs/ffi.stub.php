<?php

/**
 * Hand-rolled PHPStan stubs for ext-ffi covering the public surface
 * pdf_oxide consumes.
 *
 * Why this exists: there is no mature dedicated PHPStan extension for
 * `FFI::cdef()` as of 2026-05. The community workaround is to declare
 * stub files referenced from `phpstan.neon` `stubFiles:` (see SOTA-2026
 * report §4 "FFI type-checking"). This stub lets us type-narrow the
 * FFI return surface without refactoring the 25.6 kLoC binding.
 *
 * Stretch: wrap `FFI\CData` in an `Internal\` namespace marked
 * `@internal` so the public API trades only in PHP scalars/DTOs.
 * Once contained, PHPStan can ratchet to level 9.
 */

// phpcs:disable
// @phpcs:ignoreFile

namespace {

    /**
     * Foreign Function Interface — minimum stub for static analysis.
     *
     * @see https://www.php.net/manual/en/class.ffi.php
     */
    final class FFI
    {
        /**
         * @param string $code C declarations.
         * @param string $lib  Shared library path or name.
         */
        public static function cdef(string $code = '', ?string $lib = null): \FFI {}

        /**
         * @param string $filename Loadable C header path.
         */
        public static function load(string $filename): ?\FFI {}

        /**
         * @param string $type C type expression.
         * @param mixed $ptr   Pointer or CData.
         */
        public static function cast(string $type, mixed $ptr): \FFI\CData {}

        /**
         * @param string $type C type expression.
         * @param bool $owned  Whether the returned CData owns the memory.
         * @param bool $persistent
         */
        public function new(string $type, bool $owned = true, bool $persistent = false): \FFI\CData {}

        /**
         * Free memory associated with the given CData.
         */
        public static function free(\FFI\CData $ptr): void {}

        /**
         * @param \FFI\CData|string $source
         */
        public static function memcpy(\FFI\CData $dst, mixed $source, int $size): void {}

        /**
         * @param \FFI\CData|string $a
         * @param \FFI\CData|string $b
         */
        public static function memcmp(mixed $a, mixed $b, int $size): int {}

        /**
         * @param \FFI\CData|string $ptr
         */
        public static function string(mixed $ptr, ?int $size = null): string {}

        /**
         * @param \FFI\CData $ptr
         */
        public static function addr(\FFI\CData $ptr): \FFI\CData {}

        /**
         * @param \FFI\CData $ptr
         */
        public static function typeof(\FFI\CData $ptr): \FFI\CType {}

        public static function sizeof(\FFI\CData|\FFI\CType $ptr): int {}

        public static function alignof(\FFI\CData|\FFI\CType $ptr): int {}

        public static function isNull(\FFI\CData $ptr): bool {}

        /**
         * @param array<int, mixed> $args
         */
        public function __call(string $name, array $args): mixed {}

        public function __get(string $name): mixed {}

        public function __set(string $name, mixed $value): void {}
    }
}

namespace FFI {

    /**
     * Generic FFI C data — opaque to PHPStan without stubs.
     *
     * Treat any property access on a CData as `mixed`; the actual
     * layout comes from the cdef() string at runtime.
     *
     * @implements \ArrayAccess<int|string, mixed>
     */
    final class CData implements \ArrayAccess
    {
        public function __get(string $name): mixed {}
        public function __set(string $name, mixed $value): void {}
        /**
         * @param array<int, mixed> $args
         */
        public function __call(string $name, array $args): mixed {}

        public function offsetExists(mixed $offset): bool {}
        public function offsetGet(mixed $offset): mixed {}
        public function offsetSet(mixed $offset, mixed $value): void {}
        public function offsetUnset(mixed $offset): void {}
    }

    final class CType {}

    /**
     * Thrown by FFI::cdef() / FFI::load() / FFI::cast() on parse or
     * binding errors.
     */
    class Exception extends \Error {}

    /**
     * Thrown when an FFI operation is invoked in an invalid context.
     */
    class ParserException extends Exception {}
}
