<?php

declare(strict_types=1);

namespace PdfOxide\Exceptions;

use Exception;

/**
 * Base exception class for all PDF operations.
 *
 * Provides error context with error codes and operational details.
 */
class PdfException extends Exception
{
    private string $errorCode = '';
    private array $context = [];

    public function __construct(
        string $message = '',
        string $errorCode = '',
        array $context = [],
        int $code = 0,
        ?Exception $previous = null
    ) {
        parent::__construct($message, $code, $previous);
        $this->errorCode = $errorCode;
        $this->context = $context;
    }

    /**
     * Get the error code associated with this exception.
     */
    public function getErrorCode(): string
    {
        return $this->errorCode;
    }

    /**
     * Get the context array for this exception.
     */
    public function getContext(): array
    {
        return $this->context;
    }

    /**
     * Add context information to the exception.
     */
    public function withContext(string $key, mixed $value): self
    {
        $this->context[$key] = $value;
        return $this;
    }

    /**
     * Get formatted context string for logging.
     */
    public function getContextString(): string
    {
        if (empty($this->context)) {
            return '';
        }

        $parts = [];
        foreach ($this->context as $key => $value) {
            if (is_array($value) || is_object($value)) {
                $parts[] = "{$key}: " . json_encode($value);
            } else {
                $parts[] = "{$key}: {$value}";
            }
        }

        return ' [' . implode(', ', $parts) . ']';
    }

    /**
     * String representation of the exception.
     */
    public function __toString(): string
    {
        return sprintf(
            "%s: [%s] %s%s\nFile: %s:%d\n",
            static::class,
            $this->errorCode,
            $this->message,
            $this->getContextString(),
            $this->file,
            $this->line
        );
    }
}
