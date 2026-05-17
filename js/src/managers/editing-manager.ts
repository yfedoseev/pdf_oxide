/**
 * EditingManager - Document Editing Manager for redaction and flattening operations
 *
 * Provides PDF document editing capabilities:
 * - Content redaction (add, apply, count redaction areas)
 * - Metadata scrubbing (remove Info, XMP, JavaScript)
 * - Form flattening (all pages or single page)
 * - Annotation flattening (all pages or single page)
 *
 * Uses native FFI functions:
 * - pdf_redaction_add, pdf_redaction_apply, pdf_redaction_scrub_metadata, pdf_redaction_count
 * - pdf_document_editor_flatten_forms, pdf_document_editor_flatten_forms_page
 * - pdf_document_editor_flatten_annotations, pdf_document_editor_flatten_annotations_page
 */

import { EventEmitter } from 'events';
import { mapFfiErrorCode, PdfException } from '../errors';

// =============================================================================
// Type Definitions
// =============================================================================

/**
 * Rectangle coordinates for a redaction area.
 */
export interface RedactionRect {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/**
 * RGB color specification (values 0.0-1.0).
 */
export interface RgbColor {
  r: number;
  g: number;
  b: number;
}

/**
 * Options for applying redactions.
 */
export interface ApplyRedactionsOptions {
  /** Whether to also scrub document metadata when applying redactions. */
  scrubMetadata?: boolean;
  /** Default fill color for redacted areas. Defaults to black (0, 0, 0). */
  fillColor?: RgbColor;
}

/**
 * Options for scrubbing document metadata.
 */
export interface ScrubMetadataOptions {
  /** Remove /Info dictionary entries (Title, Author, etc.). Default: true */
  removeInfo?: boolean;
  /** Remove XMP metadata stream. Default: true */
  removeXmp?: boolean;
  /** Remove JavaScript actions. Default: true */
  removeJs?: boolean;
}

// =============================================================================
// Canonical EditingManager
// =============================================================================

/**
 * Manages document editing operations including redaction and flattening.
 *
 * @example
 * ```typescript
 * import { EditingManager } from 'pdf_oxide';
 *
 * const editor = new EditingManager(document);
 *
 * // Add redaction areas
 * editor.addRedaction(0, { x1: 100, y1: 200, x2: 300, y2: 250 });
 * editor.addRedaction(0, { x1: 50, y1: 400, x2: 200, y2: 450 }, { r: 0, g: 0, b: 0 });
 *
 * // Apply all queued redactions
 * const count = editor.applyRedactions({ scrubMetadata: true });
 * console.log(`Applied ${count} redactions`);
 *
 * // Flatten forms and annotations
 * editor.flattenForms();
 * editor.flattenAnnotations();
 * ```
 */
export class EditingManager extends EventEmitter {
  private document: any;
  private native: any;

  constructor(document: any) {
    super();
    this.document = document;
    try {
      this.native = require('../../index.node');
    } catch {
      this.native = null;
    }
  }

  // ===========================================================================
  // Redaction Operations
  // ===========================================================================

  /**
   * Adds a redaction area to the document.
   *
   * Queues a rectangular region on a page for redaction. The content within
   * the rectangle will be permanently removed when {@link applyRedactions}
   * is called.
   *
   * @param page - Zero-based page index
   * @param rect - Rectangle coordinates defining the redaction area
   * @param color - Optional fill color for the redacted area (defaults to black)
   * @throws {PdfException} If the document handle is invalid or page is out of range
   *
   * @example
   * ```typescript
   * // Redact a region with default black fill
   * editor.addRedaction(0, { x1: 100, y1: 200, x2: 300, y2: 250 });
   *
   * // Redact with a custom gray fill
   * editor.addRedaction(1, { x1: 50, y1: 100, x2: 200, y2: 150 }, { r: 0.5, g: 0.5, b: 0.5 });
   * ```
   */
  addRedaction(page: number, rect: RedactionRect, color?: RgbColor): void {
    const fillColor = color ?? { r: 0, g: 0, b: 0 };

    if (this.native?.pdf_redaction_add) {
      const errorCode = { value: 0 };
      const result = this.native.pdf_redaction_add(
        this.document?.handle ?? this.document,
        page,
        rect.x1,
        rect.y1,
        rect.x2,
        rect.y2,
        fillColor.r,
        fillColor.g,
        fillColor.b,
        errorCode
      );

      if (result < 0 && errorCode.value !== 0) {
        throw mapFfiErrorCode(errorCode.value, 'Failed to add redaction area');
      }
    } else if (this.document?.addRedaction) {
      this.document.addRedaction(page, rect, fillColor);
    } else {
      this.redactionUnavailable('addRedaction');
    }

    this.emit('redactionAdded', { page, rect, color: fillColor });
  }

  /**
   * Applies all queued redactions to the document.
   *
   * Permanently removes content within all previously added redaction
   * rectangles. Optionally scrubs document metadata and applies a
   * fill color overlay.
   *
   * @param options - Options controlling redaction behavior
   * @returns Number of redactions applied
   * @throws {PdfException} If redaction application fails
   *
   * @example
   * ```typescript
   * // Apply with metadata scrubbing
   * const count = editor.applyRedactions({ scrubMetadata: true });
   *
   * // Apply with custom fill color
   * const count = editor.applyRedactions({
   *   fillColor: { r: 1.0, g: 1.0, b: 1.0 }
   * });
   * ```
   */
  applyRedactions(options?: ApplyRedactionsOptions): number {
    const scrubMetadata = options?.scrubMetadata ?? false;
    const fillColor = options?.fillColor ?? { r: 0, g: 0, b: 0 };

    if (this.native?.pdf_redaction_apply) {
      const errorCode = { value: 0 };
      const result = this.native.pdf_redaction_apply(
        this.document?.handle ?? this.document,
        scrubMetadata,
        fillColor.r,
        fillColor.g,
        fillColor.b,
        errorCode
      );

      if (result < 0 && errorCode.value !== 0) {
        throw mapFfiErrorCode(errorCode.value, 'Failed to apply redactions');
      }

      this.emit('redactionsApplied', { count: result, scrubMetadata });
      return result;
    } else if (this.document?.applyRedactions) {
      const count = this.document.applyRedactions(options);
      this.emit('redactionsApplied', { count, scrubMetadata });
      return count ?? 0;
    }

    return this.redactionUnavailable('applyRedactions');
  }

  /**
   * Scrubs sensitive metadata from the document.
   *
   * Removes document metadata fields that may contain sensitive information,
   * such as author names, creation tools, and JavaScript.
   *
   * The underlying native sanitize (`pdf_redaction_scrub_metadata`,
   * #231) is intentionally **all-or-nothing**: it strips `/Info`, the
   * catalog XMP `/Metadata`, document JavaScript and
   * `/Names/EmbeddedFiles` together (a secret must not survive because
   * one category was opted out). A *selective* request is therefore
   * rejected fail-closed rather than silently ignored.
   *
   * @param options - per-category toggles; all must be `true`
   *   (the default) — passing any `false` throws.
   * @throws {PdfException} If a selective scrub is requested, or if
   *   metadata scrubbing fails.
   *
   * @example
   * ```typescript
   * // Remove all metadata (the only supported mode)
   * editor.scrubMetadata();
   * ```
   */
  scrubMetadata(options?: ScrubMetadataOptions): void {
    const removeInfo = options?.removeInfo ?? true;
    const removeXmp = options?.removeXmp ?? true;
    const removeJs = options?.removeJs ?? true;

    // Fail-closed: the native scrub cannot preserve a single category,
    // so refuse a partial request instead of silently doing a full
    // scrub (or pretending the toggle was honoured in the event).
    if (!removeInfo || !removeXmp || !removeJs) {
      throw new PdfException(
        '5000',
        'scrubMetadata is all-or-nothing: removeInfo/removeXmp/removeJs ' +
          'must all be true (the native sanitize strips /Info, XMP ' +
          '/Metadata, document JavaScript and /Names/EmbeddedFiles ' +
          'together). Refusing a partial scrub that could leave a ' +
          'secret behind.'
      );
    }

    if (this.native?.pdf_redaction_scrub_metadata) {
      // C ABI: pdf_redaction_scrub_metadata(handle, error_code) — a
      // full document scrub (/Info, catalog XMP /Metadata, document
      // JavaScript, /Names/EmbeddedFiles). The per-category toggles are
      // advisory; the native scrub applies the safe all-on default.
      const errorCode = { value: 0 };
      const result = this.native.pdf_redaction_scrub_metadata(
        this.document?.handle ?? this.document,
        errorCode
      );

      if (result < 0 && errorCode.value !== 0) {
        throw mapFfiErrorCode(errorCode.value, 'Failed to scrub metadata');
      }
    } else if (this.document?.scrubMetadata) {
      this.document.scrubMetadata(options);
    } else {
      this.redactionUnavailable('scrubMetadata');
    }

    this.emit('metadataScrubbed', { removeInfo, removeXmp, removeJs });
  }

  /**
   * Gets the number of redaction areas queued for a page (annotations
   * + programmatic rectangles), matching the per-page C ABI / Rust /
   * C# / Go contract (`pdf_redaction_count(handle, page, error_code)`).
   *
   * @param page - zero-based page index (default 0)
   * @returns Number of redaction areas queued for that page
   * @throws {PdfException} If the document handle is invalid
   *
   * @example
   * ```typescript
   * editor.addRedaction(0, { x1: 10, y1: 20, x2: 100, y2: 50 });
   * console.log(editor.getRedactionCount(0)); // 1
   * ```
   */
  getRedactionCount(page = 0): number {
    if (this.native?.pdf_redaction_count) {
      const errorCode = { value: 0 };
      // C ABI arity is (handle, page, error_code) — the page index is
      // mandatory; omitting it would shift errorCode into the page slot.
      const result = this.native.pdf_redaction_count(
        this.document?.handle ?? this.document,
        page,
        errorCode
      );

      if (result < 0 && errorCode.value !== 0) {
        throw mapFfiErrorCode(errorCode.value, 'Failed to get redaction count');
      }

      return result;
    } else if (this.document?.getRedactionCount) {
      return this.document.getRedactionCount(page) ?? 0;
    }

    return this.redactionUnavailable('getRedactionCount');
  }

  /**
   * Fail loudly when no real redaction backend is wired.
   *
   * Latent bug fix (README / #231): `editing-manager.ts` referenced
   * native `pdf_redaction_add/apply/scrub_metadata/count` symbols that
   * do **not** exist in `src/ffi.rs`. The previous fall-throughs
   * silently no-op'd / returned 0 / emitted a success event — a
   * security-critical "redaction" API pretending to succeed while
   * removing nothing. Destructive redaction is implemented as of
   * v0.3.50 (#231); reaching here now means the loaded native addon is
   * missing/older than the bindings and does not export the
   * `pdf_redaction_*` symbols — these operations must refuse rather
   * than give a false sense of safety.
   */
  private redactionUnavailable(op: string): never {
    throw new PdfException(
      '5000',
      `Destructive redaction is unavailable: '${op}' requires the native ` +
        `pdf_redaction_* symbols, which the loaded pdf-oxide addon does not ` +
        `export. The native module is missing or older than these bindings — ` +
        `rebuild/upgrade the native addon to match. Refusing to silently ` +
        `no-op a security-critical operation.`
    );
  }

  // ===========================================================================
  // Form Flattening Operations
  // ===========================================================================

  /**
   * Flattens all form fields in the document.
   *
   * Renders form field widgets into page content and removes the AcroForm
   * dictionary. After flattening, form fields become static content and
   * can no longer be edited interactively.
   *
   * @throws {PdfException} If flattening fails
   *
   * @example
   * ```typescript
   * editor.flattenForms();
   * ```
   */
  flattenForms(): void {
    if (this.native?.pdf_document_editor_flatten_forms) {
      const errorCode = { value: 0 };
      const result = this.native.pdf_document_editor_flatten_forms(
        this.document?.handle ?? this.document,
        errorCode
      );

      if (result < 0 && errorCode.value !== 0) {
        throw mapFfiErrorCode(errorCode.value, 'Failed to flatten forms');
      }
    } else if (this.document?.flattenForms) {
      this.document.flattenForms();
    }

    this.emit('formsFlattened');
  }

  /**
   * Flattens form fields on a specific page.
   *
   * Only flattens form field widgets located on the specified page,
   * leaving form fields on other pages editable.
   *
   * @param page - Zero-based page index
   * @throws {PdfException} If flattening fails or page is out of range
   *
   * @example
   * ```typescript
   * // Flatten forms on first page only
   * editor.flattenFormsPage(0);
   * ```
   */
  flattenFormsPage(page: number): void {
    if (this.native?.pdf_document_editor_flatten_forms_page) {
      const errorCode = { value: 0 };
      const result = this.native.pdf_document_editor_flatten_forms_page(
        this.document?.handle ?? this.document,
        page,
        errorCode
      );

      if (result < 0 && errorCode.value !== 0) {
        throw mapFfiErrorCode(errorCode.value, `Failed to flatten forms on page ${page}`);
      }
    } else if (this.document?.flattenFormsPage) {
      this.document.flattenFormsPage(page);
    }

    this.emit('formsPageFlattened', { page });
  }

  // ===========================================================================
  // Annotation Flattening Operations
  // ===========================================================================

  /**
   * Flattens all annotations in the document.
   *
   * Renders annotations into page content and removes them from
   * annotation arrays. After flattening, annotations become static
   * content and can no longer be edited or deleted.
   *
   * @throws {PdfException} If flattening fails
   *
   * @example
   * ```typescript
   * editor.flattenAnnotations();
   * ```
   */
  flattenAnnotations(): void {
    if (this.native?.pdf_document_editor_flatten_annotations) {
      const errorCode = { value: 0 };
      const result = this.native.pdf_document_editor_flatten_annotations(
        this.document?.handle ?? this.document,
        errorCode
      );

      if (result < 0 && errorCode.value !== 0) {
        throw mapFfiErrorCode(errorCode.value, 'Failed to flatten annotations');
      }
    } else if (this.document?.flattenAnnotations) {
      this.document.flattenAnnotations();
    }

    this.emit('annotationsFlattened');
  }

  /**
   * Flattens annotations on a specific page.
   *
   * Only flattens annotations located on the specified page,
   * leaving annotations on other pages editable.
   *
   * @param page - Zero-based page index
   * @throws {PdfException} If flattening fails or page is out of range
   *
   * @example
   * ```typescript
   * // Flatten annotations on page 2
   * editor.flattenAnnotationsPage(1);
   * ```
   */
  flattenAnnotationsPage(page: number): void {
    if (this.native?.pdf_document_editor_flatten_annotations_page) {
      const errorCode = { value: 0 };
      const result = this.native.pdf_document_editor_flatten_annotations_page(
        this.document?.handle ?? this.document,
        page,
        errorCode
      );

      if (result < 0 && errorCode.value !== 0) {
        throw mapFfiErrorCode(errorCode.value, `Failed to flatten annotations on page ${page}`);
      }
    } else if (this.document?.flattenAnnotationsPage) {
      this.document.flattenAnnotationsPage(page);
    }

    this.emit('annotationsPageFlattened', { page });
  }

  // ===========================================================================
  // Form Data Import/Export
  // ===========================================================================

  /**
   * Import form data from a file (FDF or XFDF, auto-detected by extension).
   * @param filePath Path to the FDF or XFDF file
   * @returns Number of fields imported
   */
  importFormDataFromFile(filePath: string): number {
    if (this.native?.pdf_document_import_form_data) {
      const errorCode = { value: 0 };
      const count = this.native.pdf_document_import_form_data(
        this.document?.handle ?? this.document,
        filePath,
        errorCode
      );
      if (errorCode.value !== 0) {
        throw mapFfiErrorCode(errorCode.value, `Failed to import form data from ${filePath}`);
      }
      return count;
    }
    throw new Error('Form data import not available');
  }

  /**
   * Import FDF form data from in-memory bytes.
   * @param data FDF data bytes
   * @returns Number of fields imported
   */
  importFdfBytes(data: Buffer): number {
    if (this.native?.pdf_editor_import_fdf_bytes) {
      const errorCode = { value: 0 };
      const count = this.native.pdf_editor_import_fdf_bytes(
        this.document?.handle ?? this.document,
        data,
        data.length,
        errorCode
      );
      if (errorCode.value !== 0) {
        throw mapFfiErrorCode(errorCode.value, 'Failed to import FDF bytes');
      }
      return count;
    }
    throw new Error('FDF byte import not available');
  }

  /**
   * Import XFDF form data from in-memory bytes.
   * @param data XFDF data bytes
   * @returns Number of fields imported
   */
  importXfdfBytes(data: Buffer): number {
    if (this.native?.pdf_editor_import_xfdf_bytes) {
      const errorCode = { value: 0 };
      const count = this.native.pdf_editor_import_xfdf_bytes(
        this.document?.handle ?? this.document,
        data,
        data.length,
        errorCode
      );
      if (errorCode.value !== 0) {
        throw mapFfiErrorCode(errorCode.value, 'Failed to import XFDF bytes');
      }
      return count;
    }
    throw new Error('XFDF byte import not available');
  }

  /**
   * Export form data to in-memory bytes.
   * @param format 0 for FDF, 1 for XFDF
   * @returns Exported form data bytes
   */
  exportFormDataToBytes(format: 0 | 1 = 0): Buffer {
    if (this.native?.pdf_document_export_form_data_to_bytes) {
      const errorCode = { value: 0 };
      const outLen = { value: 0 };
      const ptr = this.native.pdf_document_export_form_data_to_bytes(
        this.document?.handle ?? this.document,
        format,
        outLen,
        errorCode
      );
      if (errorCode.value !== 0 || !ptr) {
        throw mapFfiErrorCode(errorCode.value, 'Failed to export form data');
      }
      return Buffer.from(ptr, 0, outLen.value);
    }
    throw new Error('Form data export not available');
  }

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  /**
   * Releases resources held by this manager.
   */
  destroy(): void {
    this.removeAllListeners();
  }
}

export default EditingManager;
