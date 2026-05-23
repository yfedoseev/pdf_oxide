<?php

declare(strict_types=1);

namespace PdfOxide\Utilities;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Helper for PDF validation and integrity checking.
 *
 * Provides methods to validate PDF structure and diagnose issues.
 */
class ValidationHelper
{
    private FunctionBindings $bindings;
    private CData $handle;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Validate PDF structure.
     */
    public function validateStructure(): bool
    {
        return $this->bindings->pdfDocumentValidateStructure($this->handle);
    }

    /**
     * Get all validation errors.
     */
    public function getValidationErrors(): array
    {
        return $this->bindings->pdfDocumentGetValidationErrors($this->handle);
    }

    /**
     * Get validation report with details.
     */
    public function getValidationReport(): array
    {
        $isValid = $this->validateStructure();
        $errors = $this->getValidationErrors();

        return [
            'is_valid' => $isValid,
            'error_count' => count($errors),
            'errors' => $errors,
            'timestamp' => date('Y-m-d H:i:s'),
        ];
    }

    /**
     * Check if PDF passes all validations.
     */
    public function isValid(): bool
    {
        return $this->validateStructure() && empty($this->getValidationErrors());
    }

    /**
     * Get human-readable validation summary.
     */
    public function getSummary(): string
    {
        $report = $this->getValidationReport();
        if ($report['is_valid'] && empty($report['errors'])) {
            return 'PDF structure is valid.';
        }

        $summary = sprintf("PDF has %d validation errors:\n", $report['error_count']);
        foreach ($report['errors'] as $error) {
            if (is_array($error)) {
                $summary .= sprintf("  - [%s] %s\n", $error['code'] ?? 'ERR', $error['message'] ?? 'Unknown error');
            } else {
                $summary .= sprintf("  - %s\n", $error);
            }
        }

        return $summary;
    }

    /**
     * Throw exception if validation fails.
     */
    public function assertValid(): void
    {
        if (!$this->isValid()) {
            throw new \RuntimeException("PDF validation failed:\n" . $this->getSummary());
        }
    }
}
