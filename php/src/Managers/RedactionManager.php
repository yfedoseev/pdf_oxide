<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\Exceptions\RedactionException;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\Rect;

/**
 * v0.3.50 #231 — true destructive redaction.
 *
 * Wraps the six C-ABI redaction functions
 * (`pdf_redaction_add`, `pdf_redaction_count`, `pdf_redaction_apply`,
 * `pdf_redaction_scrub_metadata`, `document_editor_apply_page_redactions`,
 * `document_editor_apply_all_redactions`).
 *
 * Why a dedicated manager (instead of folding into
 * {@see DocumentEditorManager}): redaction is a SECURITY OP per
 * `feedback_extraction_graceful_fallback` — every error path here
 * fails closed (throws). Isolating the API surface makes that
 * contract obvious at the call site.
 *
 * Construct from a path or from an existing `DocumentEditor*` handle:
 *
 *   $redact = RedactionManager::openFile('/tmp/in.pdf');
 *   $redact->mark(0, new Rect(100, 100, 200, 50));
 *   $redact->apply();
 *
 * The redaction ABI accepts a `DocumentEditor*`, NOT a `PdfDocument*` —
 * a `DocumentEditor` is the writable counterpart (`document_editor_open`)
 * versus the read-only `pdf_document_open`.
 */
final class RedactionManager
{
    private readonly FunctionBindings $bindings;
    private readonly CData $editorHandle;
    private bool $ownsHandle;

    /**
     * @param CData $editorHandle a `DocumentEditor*` handle.
     * @param bool $ownsHandle whether {@see __destruct()} should free it.
     */
    public function __construct(CData $editorHandle, bool $ownsHandle = false)
    {
        $this->editorHandle = $editorHandle;
        $this->bindings = new FunctionBindings();
        $this->ownsHandle = $ownsHandle;
    }

    /**
     * Open a PDF file for redaction. Returns a manager that OWNS the
     * underlying DocumentEditor handle (freed on destruct).
     */
    public static function openFile(string $path): self
    {
        $bindings = new FunctionBindings();
        $handle = $bindings->documentEditorOpen($path);
        if ($handle === null) {
            throw new RedactionException("Failed to open document editor for: {$path}");
        }
        return new self($handle, ownsHandle: true);
    }

    public function __destruct()
    {
        if ($this->ownsHandle) {
            try {
                $this->bindings->documentEditorFree($this->editorHandle);
            } catch (\Throwable) {
                // Best-effort; PHP shutdown already may have torn down FFI.
            }
        }
    }

    /**
     * Mark a single rectangle for destructive redaction on a page.
     *
     * @param int $pageIndex zero-based.
     * @param Rect $rect coordinates in PDF points.
     * @param array{0:float,1:float,2:float}|null $color RGB 0–1 fill; null = black.
     */
    public function mark(int $pageIndex, Rect $rect, ?array $color = null): void
    {
        [$r, $g, $b] = $color ?? [0.0, 0.0, 0.0];
        $this->bindings->pdfRedactionAdd(
            $this->editorHandle,
            $pageIndex,
            $rect->x,
            $rect->y,
            $rect->x + $rect->width,
            $rect->y + $rect->height,
            $r,
            $g,
            $b,
        );
    }

    /**
     * Mark several rectangles in one call.
     *
     * @param array<int,Rect> $rects
     * @param array{0:float,1:float,2:float}|null $color RGB 0–1.
     */
    public function markRects(int $pageIndex, array $rects, ?array $color = null): void
    {
        foreach ($rects as $rect) {
            $this->mark($pageIndex, $rect, $color);
        }
    }

    /** Count pending redaction marks on a page. */
    public function pendingCount(int $pageIndex): int
    {
        return $this->bindings->pdfRedactionCount($this->editorHandle, $pageIndex);
    }

    /**
     * Apply ALL pending redactions destructively (byte-level scrub).
     *
     * @param bool $scrubMetadata also wipe the document metadata.
     * @param array{0:float,1:float,2:float}|null $color RGB 0–1 fill for redacted regions.
     * @throws RedactionException on any non-zero ABI error code.
     */
    public function apply(bool $scrubMetadata = true, ?array $color = null): void
    {
        [$r, $g, $b] = $color ?? [0.0, 0.0, 0.0];
        try {
            $this->bindings->pdfRedactionApply(
                $this->editorHandle,
                $scrubMetadata,
                $r,
                $g,
                $b,
            );
        } catch (\PdfOxide\Exceptions\PdfException $e) {
            // SECURITY-OP contract: convert ANY error class to RedactionException
            // so callers can fail-closed on a single catch.
            if ($e instanceof RedactionException) {
                throw $e;
            }
            throw new RedactionException(
                'Redaction apply failed: ' . $e->getMessage(),
                $e->getContext(),
                $e
            );
        }
    }

    /** Apply redactions for a single page only. */
    public function applyPage(int $pageIndex): void
    {
        try {
            $this->bindings->documentEditorApplyPageRedactions($this->editorHandle, $pageIndex);
        } catch (\PdfOxide\Exceptions\PdfException $e) {
            if ($e instanceof RedactionException) {
                throw $e;
            }
            throw new RedactionException(
                'Per-page redaction apply failed: ' . $e->getMessage(),
                $e->getContext() + ['page' => $pageIndex],
                $e
            );
        }
    }

    /** Apply redactions across every marked page (convenience). */
    public function applyAll(): void
    {
        try {
            $this->bindings->documentEditorApplyAllRedactions($this->editorHandle);
        } catch (\PdfOxide\Exceptions\PdfException $e) {
            if ($e instanceof RedactionException) {
                throw $e;
            }
            throw new RedactionException(
                'Apply-all redactions failed: ' . $e->getMessage(),
                $e->getContext(),
                $e
            );
        }
    }

    /**
     * Destructively wipe all document metadata
     * (Info dict, XMP, etc.). Independent of any rect redactions.
     *
     * @throws RedactionException on failure.
     */
    public function scrubMetadata(): void
    {
        try {
            $this->bindings->pdfRedactionScrubMetadata($this->editorHandle);
        } catch (\PdfOxide\Exceptions\PdfException $e) {
            if ($e instanceof RedactionException) {
                throw $e;
            }
            throw new RedactionException(
                'Metadata scrub failed: ' . $e->getMessage(),
                $e->getContext(),
                $e
            );
        }
    }
}
