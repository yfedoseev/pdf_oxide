<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Manages PDF document metadata and properties.
 *
 * Provides access to document information, security permissions, and structural details.
 */
class MetadataManager
{
    private FunctionBindings $bindings;
    private CData $handle;
    private array $cachedMetadata = [];

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    // ==================== BASIC METADATA ====================

    /**
     * Get document title.
     */
    public function getTitle(): string
    {
        if (!isset($this->cachedMetadata['title'])) {
            $this->cachedMetadata['title'] = $this->bindings->pdfDocumentGetTitle($this->handle);
        }
        return $this->cachedMetadata['title'];
    }

    /**
     * Get document author.
     */
    public function getAuthor(): string
    {
        if (!isset($this->cachedMetadata['author'])) {
            $this->cachedMetadata['author'] = $this->bindings->pdfDocumentGetAuthor($this->handle);
        }
        return $this->cachedMetadata['author'];
    }

    /**
     * Get document subject.
     */
    public function getSubject(): string
    {
        if (!isset($this->cachedMetadata['subject'])) {
            $this->cachedMetadata['subject'] = $this->bindings->pdfDocumentGetSubject($this->handle);
        }
        return $this->cachedMetadata['subject'];
    }

    /**
     * Get document keywords.
     */
    public function getKeywords(): string
    {
        if (!isset($this->cachedMetadata['keywords'])) {
            $this->cachedMetadata['keywords'] = $this->bindings->pdfDocumentGetKeywords($this->handle);
        }
        return $this->cachedMetadata['keywords'];
    }

    /**
     * Get creator application.
     */
    public function getCreator(): string
    {
        if (!isset($this->cachedMetadata['creator'])) {
            $this->cachedMetadata['creator'] = $this->bindings->pdfDocumentGetCreator($this->handle);
        }
        return $this->cachedMetadata['creator'];
    }

    /**
     * Get producer application.
     */
    public function getProducer(): string
    {
        if (!isset($this->cachedMetadata['producer'])) {
            $this->cachedMetadata['producer'] = $this->bindings->pdfDocumentGetProducer($this->handle);
        }
        return $this->cachedMetadata['producer'];
    }

    /**
     * Get document creation date (ISO 8601 format).
     */
    public function getCreationDate(): string
    {
        if (!isset($this->cachedMetadata['creation_date'])) {
            $this->cachedMetadata['creation_date'] = $this->bindings->pdfDocumentGetCreationDate($this->handle);
        }
        return $this->cachedMetadata['creation_date'];
    }

    /**
     * Get document modification date (ISO 8601 format).
     */
    public function getModificationDate(): string
    {
        if (!isset($this->cachedMetadata['mod_date'])) {
            $this->cachedMetadata['mod_date'] = $this->bindings->pdfDocumentGetModDate($this->handle);
        }
        return $this->cachedMetadata['mod_date'];
    }

    // ==================== SECURITY & PERMISSIONS ====================

    /**
     * Check if document is encrypted.
     */
    public function isEncrypted(): bool
    {
        return $this->bindings->pdfDocumentIsEncrypted($this->handle);
    }

    /**
     * Get encryption algorithm (e.g., "RC4", "AES", "AES-256").
     */
    public function getEncryptionAlgorithm(): string
    {
        if (!isset($this->cachedMetadata['encryption_algorithm'])) {
            $this->cachedMetadata['encryption_algorithm'] = $this->bindings->pdfDocumentGetEncryptionAlgorithm($this->handle);
        }
        return $this->cachedMetadata['encryption_algorithm'];
    }

    /**
     * Check if document requires a password for opening.
     */
    public function requiresPassword(): bool
    {
        return $this->bindings->pdfDocumentRequiresPassword($this->handle);
    }

    /**
     * Check if printing is allowed.
     */
    public function canPrint(): bool
    {
        if (!isset($this->cachedMetadata['can_print'])) {
            $this->cachedMetadata['can_print'] = $this->bindings->pdfDocumentCanPrint($this->handle);
        }
        return $this->cachedMetadata['can_print'];
    }

    /**
     * Check if copying content is allowed.
     */
    public function canCopy(): bool
    {
        if (!isset($this->cachedMetadata['can_copy'])) {
            $this->cachedMetadata['can_copy'] = $this->bindings->pdfDocumentCanCopy($this->handle);
        }
        return $this->cachedMetadata['can_copy'];
    }

    /**
     * Check if modifying document is allowed.
     */
    public function canModify(): bool
    {
        if (!isset($this->cachedMetadata['can_modify'])) {
            $this->cachedMetadata['can_modify'] = $this->bindings->pdfDocumentCanModify($this->handle);
        }
        return $this->cachedMetadata['can_modify'];
    }

    /**
     * Check if filling forms is allowed.
     */
    public function canFillForms(): bool
    {
        if (!isset($this->cachedMetadata['can_fill_forms'])) {
            $this->cachedMetadata['can_fill_forms'] = $this->bindings->pdfDocumentCanFillForms($this->handle);
        }
        return $this->cachedMetadata['can_fill_forms'];
    }

    /**
     * Check if annotations are allowed.
     */
    public function canAnnotate(): bool
    {
        if (!isset($this->cachedMetadata['can_annotate'])) {
            $this->cachedMetadata['can_annotate'] = $this->bindings->pdfDocumentCanAnnotate($this->handle);
        }
        return $this->cachedMetadata['can_annotate'];
    }

    /**
     * Get permission summary.
     */
    public function getPermissions(): array
    {
        return [
            'can_print' => $this->canPrint(),
            'can_copy' => $this->canCopy(),
            'can_modify' => $this->canModify(),
            'can_fill_forms' => $this->canFillForms(),
            'can_annotate' => $this->canAnnotate(),
        ];
    }

    /**
     * Check if document is protected (any permission restriction).
     */
    public function isProtected(): bool
    {
        return $this->isEncrypted() ||
               !$this->canPrint() ||
               !$this->canCopy() ||
               !$this->canModify() ||
               !$this->canFillForms() ||
               !$this->canAnnotate();
    }

    // ==================== STRUCTURAL PROPERTIES ====================

    /**
     * Check if document is a tagged PDF (has structure tree).
     */
    public function isTaggedPdf(): bool
    {
        if (!isset($this->cachedMetadata['is_tagged_pdf'])) {
            $this->cachedMetadata['is_tagged_pdf'] = $this->bindings->pdfDocumentHasStructureTree($this->handle);
        }
        return $this->cachedMetadata['is_tagged_pdf'];
    }

    /**
     * Get all metadata as array.
     */
    public function toArray(): array
    {
        return [
            'title' => $this->getTitle(),
            'author' => $this->getAuthor(),
            'subject' => $this->getSubject(),
            'keywords' => $this->getKeywords(),
            'creator' => $this->getCreator(),
            'producer' => $this->getProducer(),
            'creation_date' => $this->getCreationDate(),
            'modification_date' => $this->getModificationDate(),
            'is_encrypted' => $this->isEncrypted(),
            'encryption_algorithm' => $this->getEncryptionAlgorithm(),
            'requires_password' => $this->requiresPassword(),
            'permissions' => $this->getPermissions(),
            'is_protected' => $this->isProtected(),
            'is_tagged_pdf' => $this->isTaggedPdf(),
        ];
    }

    /**
     * Clear cached metadata.
     *
     * @internal
     */
    public function clearCache(): void
    {
        $this->cachedMetadata = [];
    }
}
