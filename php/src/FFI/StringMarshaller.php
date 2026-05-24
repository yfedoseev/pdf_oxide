<?php

declare(strict_types=1);

namespace PdfOxide\FFI;

use FFI;
use FFI\CData;

/**
 * Handles marshaling of strings between PHP and C/FFI.
 *
 * Manages UTF-8 encoding/decoding and memory management.
 */
class StringMarshaller
{
    /**
     * Convert a PHP string to a C string (char*) for FFI.
     *
     * @param string $str The PHP string to convert
     * @return CData FFI CData representation of the C string
     */
    public static function toCString(string $str): CData
    {
        $ffi = NativeLibrary::getInstance();
        $bytes = strlen($str) + 1;

        // Allocate C string
        $cStr = FFI::new('char[' . $bytes . ']');

        // Copy bytes
        FFI::memcpy($cStr, $str, strlen($str));
        $cStr[strlen($str)] = "\0";

        return $cStr;
    }

    /**
     * Convert a C string (char*) to a PHP string.
     *
     * @param CData|null $cStr The C string pointer (may be a null pointer
     *                          when the native side has nothing to return).
     * @param bool $free Whether to free the C memory after conversion
     * @return string The PHP string
     */
    public static function fromCString(?CData $cStr, bool $free = true): string
    {
        if ($cStr === null) {
            return '';
        }

        // Use FFI::string() to copy the NUL-terminated C string in
        // O(n) rather than concatenating char-by-char in O(n²) — for
        // large extracted-text / markdown buffers the quadratic form
        // dominated wall time. The no-length overload reads until NUL
        // automatically; ext-ffi handles the strlen in C.
        $str = FFI::string(FFI::cast('char*', $cStr));

        if (!self::isValidUtf8($str)) {
            throw new \RuntimeException('Invalid UTF-8 string from FFI');
        }

        // Free C memory if requested
        if ($free) {
            self::freeString($cStr);
        }

        return $str;
    }

    /**
     * Free a C string allocated by the native library.
     *
     * @param CData|null $cStr The C string to free
     */
    public static function freeString(?CData $cStr): void
    {
        if ($cStr === null) {
            return;
        }

        try {
            $ffi = NativeLibrary::getInstance();
            $ffi->free_string(FFI::cast('char*', $cStr));
        } catch (\Exception $e) {
            // Log but don't throw - we're in cleanup
            trigger_error('Failed to free C string: ' . $e->getMessage(), E_USER_WARNING);
        }
    }

    /**
     * Check if a string is valid UTF-8. Used internally by toCString.
     */
    private static function isValidUtf8(string $str): bool
    {
        return mb_check_encoding($str, 'UTF-8');
    }
}
