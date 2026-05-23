<?php

declare(strict_types=1);

namespace PdfOxide\Types;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Represents PDF compliance validation results.
 *
 * Contains compliance status, errors, and warnings for standards like PDF/A, PDF/X, PDF/UA.
 */
class ComplianceResult
{
    private CData $handle;
    private string $standard;
    private FunctionBindings $bindings;
    private string $resultType; // 'pdfPdfA', 'pdfPdfX', or 'pdfPdfUa'
    private ?bool $cachedCompliance = null;
    private ?array $cachedErrors = null;
    private ?array $cachedWarnings = null;

    public function __construct(CData $handle, string $standard, FunctionBindings $bindings, string $resultType)
    {
        $this->handle = $handle;
        $this->standard = $standard;
        $this->bindings = $bindings;
        $this->resultType = $resultType;
    }

    /**
     * Get compliance standard.
     *
     * @return string Standard name (PDF/A-A2B, PDF/X-4, PDF/UA, etc.)
     */
    public function getStandard(): string
    {
        return $this->standard;
    }

    /**
     * Check if document is compliant.
     *
     * @return bool True if compliant
     */
    public function isCompliant(): bool
    {
        if ($this->cachedCompliance === null) {
            if ($this->resultType === 'pdfPdfA') {
                $this->cachedCompliance = $this->bindings->pdfPdfAIsCompliant($this->handle);
            } elseif ($this->resultType === 'pdfPdfX') {
                $this->cachedCompliance = $this->bindings->pdfPdfXIsCompliant($this->handle);
            } else { // pdfPdfUa
                $this->cachedCompliance = $this->bindings->pdfPdfUaIsAccessible($this->handle);
            }
        }
        return $this->cachedCompliance;
    }

    /**
     * Get error count.
     *
     * @return int Number of compliance errors
     */
    public function getErrorCount(): int
    {
        if ($this->resultType === 'pdfPdfA') {
            return $this->bindings->pdfPdfAErrorCount($this->handle);
        } elseif ($this->resultType === 'pdfPdfX') {
            return $this->bindings->pdfPdfXErrorCount($this->handle);
        } else { // pdfPdfUa
            return $this->bindings->pdfPdfUaErrorCount($this->handle);
        }
    }

    /**
     * Get warning count.
     *
     * @return int Number of warnings (only for PDF/A)
     */
    public function getWarningCount(): int
    {
        if ($this->resultType === 'pdfPdfA') {
            return $this->bindings->pdfPdfAWarningCount($this->handle);
        }
        return 0;
    }

    /**
     * Get errors.
     *
     * @return string[] Array of error messages
     */
    public function getErrors(): array
    {
        if ($this->cachedErrors === null) {
            $this->cachedErrors = [];
            $count = $this->getErrorCount();
            for ($i = 0; $i < $count; $i++) {
                if ($this->resultType === 'pdfPdfA') {
                    $this->cachedErrors[] = $this->bindings->pdfPdfAGetError($this->handle, $i);
                } elseif ($this->resultType === 'pdfPdfX') {
                    // PDF/X doesn't have individual error retrieval in all implementations
                    $this->cachedErrors[] = 'Error ' . ($i + 1);
                } else { // pdfPdfUa
                    $this->cachedErrors[] = 'Accessibility issue ' . ($i + 1);
                }
            }
        }
        return $this->cachedErrors;
    }

    /**
     * Get warnings.
     *
     * @return string[] Array of warning messages (only for PDF/A)
     */
    public function getWarnings(): array
    {
        if ($this->cachedWarnings === null) {
            $this->cachedWarnings = [];
            if ($this->resultType === 'pdfPdfA') {
                $count = $this->getWarningCount();
                for ($i = 0; $i < $count; $i++) {
                    $this->cachedWarnings[] = $this->bindings->pdfPdfAGetWarning($this->handle, $i);
                }
            }
        }
        return $this->cachedWarnings;
    }

    /**
     * Get full validation report.
     *
     * @return string Detailed report (only for PDF/A)
     */
    public function getReport(): string
    {
        if ($this->resultType === 'pdfPdfA') {
            return $this->bindings->pdfPdfAGetReport($this->handle);
        }
        return '';
    }

    /**
     * Check if there are any issues.
     *
     * @return bool True if there are errors or warnings
     */
    public function hasIssues(): bool
    {
        return $this->getErrorCount() > 0 || $this->getWarningCount() > 0;
    }

    /**
     * Get compliance summary.
     *
     * @return array Summary information
     */
    public function getSummary(): array
    {
        return [
            'standard' => $this->standard,
            'compliant' => $this->isCompliant(),
            'errors' => $this->getErrorCount(),
            'warnings' => $this->getWarningCount(),
            'has_issues' => $this->hasIssues(),
        ];
    }

    /**
     * Get all issues as array.
     *
     * @return array All errors and warnings combined
     */
    public function getAllIssues(): array
    {
        return [
            'errors' => $this->getErrors(),
            'warnings' => $this->getWarnings(),
        ];
    }

    /**
     * Convert to array representation.
     *
     * @return array Full result data
     */
    public function toArray(): array
    {
        return [
            'standard' => $this->standard,
            'compliant' => $this->isCompliant(),
            'summary' => $this->getSummary(),
            'issues' => $this->getAllIssues(),
            'report' => $this->getReport(),
        ];
    }

    /**
     * Free result resources.
     */
    public function __destruct()
    {
        if ($this->resultType === 'pdfPdfA') {
            $this->bindings->pdfPdfAResultFree($this->handle);
        } elseif ($this->resultType === 'pdfPdfX') {
            $this->bindings->pdfPdfXResultFree($this->handle);
        } else { // pdfPdfUa
            $this->bindings->pdfPdfUaResultFree($this->handle);
        }
    }
}
