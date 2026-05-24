<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

use Exception;

/**
 * Thrown when the cdylib was built without support for the
 * requested operation (cdylib error code 8 / `_ERR_UNSUPPORTED`).
 * Mirrors the C# binding's
 * `PdfOxide.Exceptions.UnsupportedFeatureException`, the Ruby
 * binding's `PdfOxide::UnsupportedFeatureError`, and the Java
 * binding's `fyi.oxide.pdf.exception.PdfUnsupportedException`.
 *
 * <p>Common causes: cdylib compiled without `--features signatures`
 * / `ocr` / `barcodes` / `rendering`. The composer-installed
 * libpdf_oxide-vX.Y.Z-* releases ship with all features enabled;
 * users hitting this in production typically have a hand-built
 * cdylib path forced via {@code PDF_OXIDE_CDYLIB_PATH}.
 */
class UnsupportedException extends PdfException
{
    public function __construct(
        string $message = 'Operation not supported by this cdylib build',
        array $context = [],
        ?Exception $previous = null
    ) {
        parent::__construct($message, 'UNSUPPORTED', $context, 8, $previous);
    }
}
