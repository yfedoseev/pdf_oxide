<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI;
use FFI\CData;
use PdfOxide\FFI\NativeLibrary;
use PdfOxide\FFI\ErrorHandler;
use PdfOxide\FFI\StringMarshaller;

/**
 * Manages PDF accessibility operations.
 *
 * Provides structure tree inspection, automatic tagging, and
 * accessibility metadata management for PDF/UA compliance.
 *
 * Example:
 *     $manager = new AccessibilityManager($documentHandle);
 *     if ($manager->isTagged()) {
 *         $tree = $manager->getStructureTree();
 *     }
 *     $result = $manager->autoTag('en-US');
 */
class AccessibilityManager
{
    private readonly CData $handle;
    private readonly FFI $ffi;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->ffi = NativeLibrary::getInstance();
    }

    // ==================== STRUCTURE INSPECTION ====================

    /**
     * Check if the document has a structure tree (is tagged).
     *
     * @return bool True if the document contains a structure tree
     */
    public function isTagged(): bool
    {
        $errorCode = FFI::new('int');
        $result = $this->ffi->pdf_accessibility_is_tagged(
            $this->handle,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_is_tagged');
        return (bool)$result;
    }

    /**
     * Get the document's structure tree.
     *
     * @return StructureTree|null StructureTree if the document is tagged, null otherwise
     */
    public function getStructureTree(): ?StructureTree
    {
        if (!$this->isTagged()) {
            return null;
        }

        $errorCode = FFI::new('int');
        $treeHandle = $this->ffi->pdf_accessibility_get_structure_tree(
            $this->handle,
            FFI::addr($errorCode)
        );
        ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_get_structure_tree');

        return new StructureTree($treeHandle, $this->ffi);
    }

    // ==================== AUTO-TAGGING ====================

    /**
     * Automatically tag the document for accessibility.
     *
     * Analyzes content and generates a structure tree with paragraphs,
     * headings, lists, and other semantic elements.
     *
     * @param string|null $language BCP 47 language tag (e.g., "en-US"). Optional.
     * @return AutoTagResult Result with the number of elements tagged
     * @throws \PdfOxide\Exceptions\AccessibilityException If tagging fails
     */
    public function autoTag(?string $language = null): AutoTagResult
    {
        $cLanguage = $language !== null ? StringMarshaller::toCString($language) : null;
        $errorCode = FFI::new('int');

        try {
            $elementsTagged = $this->ffi->pdf_accessibility_auto_tag(
                $this->handle,
                $cLanguage,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_auto_tag');

            return new AutoTagResult(
                elementsTagged: (int)$elementsTagged
            );
        } finally {
            unset($cLanguage);
        }
    }

    // ==================== METADATA ====================

    /**
     * Set alternate text on a structure element.
     *
     * Alt text is required for non-text content in PDF/UA.
     *
     * @param int $page Page index (0-based)
     * @param int $mcid Marked content ID
     * @param string $text Alt text string
     * @throws \PdfOxide\Exceptions\AccessibilityException If the operation fails
     */
    public function setAltText(int $page, int $mcid, string $text): void
    {
        $cText = StringMarshaller::toCString($text);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_accessibility_set_alt_text(
                $this->handle,
                $page,
                $mcid,
                $cText,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_set_alt_text', [
                'page' => $page,
                'mcid' => $mcid,
            ]);
        } finally {
            unset($cText);
        }
    }

    /**
     * Set the document language.
     *
     * @param string $language BCP 47 language tag (e.g., "en-US")
     * @throws \PdfOxide\Exceptions\AccessibilityException If the operation fails
     */
    public function setLanguage(string $language): void
    {
        $cLanguage = StringMarshaller::toCString($language);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_accessibility_set_language(
                $this->handle,
                $cLanguage,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_set_language');
        } finally {
            unset($cLanguage);
        }
    }

    /**
     * Set the document title for accessibility.
     *
     * @param string $title Document title
     * @throws \PdfOxide\Exceptions\AccessibilityException If the operation fails
     */
    public function setTitle(string $title): void
    {
        $cTitle = StringMarshaller::toCString($title);
        $errorCode = FFI::new('int');

        try {
            $this->ffi->pdf_accessibility_set_title(
                $this->handle,
                $cTitle,
                FFI::addr($errorCode)
            );
            ErrorHandler::check($errorCode->cdata, 'pdf_accessibility_set_title');
        } finally {
            unset($cTitle);
        }
    }

    // ==================== SUMMARY ====================

    /**
     * Get accessibility capabilities summary.
     *
     * @return array Summary of accessibility state and capabilities
     */
    public function getSummary(): array
    {
        return [
            'is_tagged' => $this->isTagged(),
            'capabilities' => [
                'auto_tag' => true,
                'structure_tree' => true,
                'alt_text' => true,
                'language' => true,
                'title' => true,
            ],
        ];
    }
}

// ==================== SUPPORTING CLASSES ====================

/**
 * Represents a structure element in the PDF structure tree.
 */
class StructureElement
{
    public function __construct(
        public readonly string $structType,
        public readonly ?string $altText = null,
        /** @var array<StructureElement> */
        public readonly array $children = []
    ) {}

    public function toArray(): array
    {
        return [
            'struct_type' => $this->structType,
            'alt_text' => $this->altText,
            'children' => array_map(fn($c) => $c->toArray(), $this->children),
        ];
    }
}

/**
 * Represents a PDF structure tree root.
 */
class StructureTree
{
    private CData $handle;
    private FFI $ffi;

    /** @var array<StructureElement> */
    private array $rootElements = [];

    public function __construct(CData $handle, FFI $ffi)
    {
        $this->handle = $handle;
        $this->ffi = $ffi;
    }

    /**
     * Get the total number of root elements.
     */
    public function getElementCount(): int
    {
        return count($this->rootElements);
    }

    /**
     * Get root elements.
     *
     * @return array<StructureElement>
     */
    public function getRootElements(): array
    {
        return $this->rootElements;
    }

    public function __destruct()
    {
        if (isset($this->handle)) {
            $this->ffi->pdf_structure_tree_free($this->handle);
        }
    }
}

/**
 * Result of automatic document tagging.
 */
readonly class AutoTagResult
{
    public function __construct(
        public int $elementsTagged
    ) {}

    public function toArray(): array
    {
        return [
            'elements_tagged' => $this->elementsTagged,
        ];
    }
}
