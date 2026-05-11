/**
 * PDF Oxide - Complete PDF toolkit for Node.js/TypeScript
 *
 * TypeScript definitions for the pdf_oxide native module
 * Provides complete PDF manipulation, extraction, and creation capabilities
 */

// ============================================================================
// Version Functions
// ============================================================================

/**
 * Gets the Node.js binding version
 */
export function getVersion(): string;

/**
 * Gets the underlying pdf_oxide library version
 */
export function getPdfOxideVersion(): string;

/** Returns the active crypto provider name: "rust-crypto" (default) or "aws-lc-rs" (FIPS). */
export function getActiveCryptoProvider(): string;

/** Returns true if the FIPS-validated aws-lc-rs provider was compiled into this build. */
export function isFipsCryptoAvailable(): boolean;

/**
 * Installs the FIPS-validated aws-lc-rs provider as the process-wide active backend.
 * Throws if the FIPS provider was not compiled in (build without --features fips).
 */
export function useFipsCryptoProvider(): void;

// ============================================================================
// Error Classes
// ============================================================================

/**
 * Base class for all PDF-related errors
 */
export class PdfError extends Error {
  constructor(message: string);
}

/**
 * Thrown when an I/O operation fails (file read/write, etc.)
 */
export class PdfIoError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when PDF parsing fails
 */
export class PdfParseError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when PDF encryption/decryption fails
 */
export class PdfEncryptionError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when an unsupported PDF feature is encountered
 */
export class PdfUnsupportedError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when the PDF is in an invalid state for the requested operation
 */
export class PdfInvalidStateError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when PDF content decoding fails
 */
export class PdfDecodeError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when PDF content encoding fails
 */
export class PdfEncodeError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when font operations fail
 */
export class PdfFontError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when image operations fail
 */
export class PdfImageError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when a circular reference is detected in PDF structure
 */
export class PdfCircularReferenceError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when recursion limit is exceeded
 */
export class PdfRecursionLimitError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when OCR operations fail (requires 'ocr' feature)
 */
export class PdfOcrError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when machine learning operations fail (requires 'ml' feature)
 */
export class PdfMlError extends PdfError {
  constructor(message: string);
}

/**
 * Thrown when barcode operations fail
 */
export class PdfBarcodeError extends PdfError {
  constructor(message: string);
}

/**
 * Alias for PdfError for compatibility
 */
export const PdfException: typeof PdfError;

// ============================================================================
// Error Utilities
// ============================================================================

/**
 * Wraps a native error object into the appropriate JavaScript Error subclass
 * Converts native error codes to proper instanceof-compatible Error instances
 *
 * @param error - The error from native binding
 * @returns The wrapped Error instance
 *
 * @example
 * ```typescript
 * try {
 *   const doc = PdfDocument.open('file.pdf');
 * } catch (nativeErr) {
 *   const err = wrapError(nativeErr);
 *   if (err instanceof PdfIoError) {
 *     console.log('File I/O error:', err.message);
 *   }
 * }
 * ```
 */
export function wrapError(error: Error | object | string): PdfError;

/**
 * Creates a method wrapper that catches native errors and converts them to proper JavaScript Error subclasses
 *
 * @param fn - The native method to wrap
 * @param thisArg - The context (this) to bind the method to
 * @returns Wrapped function with error conversion
 *
 * @example
 * ```typescript
 * const wrappedMethod = wrapMethod(nativeMethod, this);
 * try {
 *   const result = wrappedMethod('arg1', 'arg2');
 * } catch (err) {
 *   // err is now a proper JavaScript Error subclass
 * }
 * ```
 */
export function wrapMethod(
  fn: (...args: unknown[]) => unknown,
  thisArg?: unknown
): (...args: unknown[]) => unknown;

/**
 * Creates an async method wrapper that catches native errors and converts them to proper JavaScript Error subclasses
 *
 * @param fn - The async native method to wrap
 * @param thisArg - The context (this) to bind the method to
 * @returns Wrapped async function with error conversion
 *
 * @example
 * ```typescript
 * const wrappedAsync = wrapAsyncMethod(nativeAsyncMethod, this);
 * try {
 *   const result = await wrappedAsync('arg1', 'arg2');
 * } catch (err) {
 *   // err is now a proper JavaScript Error subclass
 * }
 * ```
 */
export function wrapAsyncMethod(
  fn: (...args: unknown[]) => Promise<unknown>,
  thisArg?: unknown
): (...args: unknown[]) => Promise<unknown>;

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Represents a page size (width, height in points)
 */
export class Rect {
  x: number;
  y: number;
  width: number;
  height: number;

  constructor(x: number, y: number, width: number, height: number);

  /**
   * Gets the right edge coordinate
   */
  getRight(): number;

  /**
   * Gets the bottom edge coordinate
   */
  getBottom(): number;

  /**
   * Checks if this rect contains another point
   */
  contains(x: number, y: number): boolean;

  /**
   * Checks if this rect intersects with another rect
   */
  intersects(other: Rect): boolean;
}

/**
 * Represents a 2D point (x, y)
 */
export class Point {
  x: number;
  y: number;

  constructor(x: number, y: number);

  /**
   * Gets the distance to another point
   */
  distanceTo(other: Point): number;
}

/**
 * Represents a color in RGB
 */
export class Color {
  red: number;
  green: number;
  blue: number;

  constructor(red: number, green: number, blue: number);

  /**
   * Creates a color from hex string (#RRGGBB)
   */
  static fromHex(hex: string): Color;

  /**
   * Converts to hex string (#RRGGBB)
   */
  toHex(): string;
}

/**
 * Represents a standard page size
 */
export enum PageSize {
  Letter = 'Letter',
  Legal = 'Legal',
  A0 = 'A0',
  A1 = 'A1',
  A2 = 'A2',
  A3 = 'A3',
  A4 = 'A4',
  A5 = 'A5',
  A6 = 'A6',
  B4 = 'B4',
  B5 = 'B5',
  B6 = 'B6',
  Tabloid = 'Tabloid',
  Ledger = 'Ledger',
  Custom = 'Custom',
}

/**
 * Options for text search
 */
export interface SearchOptions {
  /**
   * Whether search is case-sensitive (default: false)
   */
  caseSensitive?: boolean;

  /**
   * Whether to match whole words only (default: false)
   */
  wholeWordsOnly?: boolean;

  /**
   * Whether to use regex matching (default: false)
   */
  useRegex?: boolean;

  /**
   * Starting page index (default: 0)
   */
  startPage?: number;

  /**
   * Ending page index (default: last page)
   */
  endPage?: number;
}

/**
 * Result of a text search
 */
export interface SearchResult {
  /**
   * The page index where text was found (0-based)
   */
  pageIndex: number;

  /**
   * The matched text
   */
  text: string;

  /**
   * Position in page content
   */
  position: number;

  /**
   * Bounding box of the match
   */
  bounds?: Rect;
}

/**
 * Options for content conversion
 */
export interface ConversionOptions {
  /**
   * Preserve formatting when converting
   */
  preserveFormatting?: boolean;

  /**
   * Include tables in conversion
   */
  includeTables?: boolean;

  /**
   * Include images in conversion
   */
  includeImages?: boolean;

  /**
   * Extract headings
   */
  extractHeadings?: boolean;

  /**
   * Extract lists
   */
  extractLists?: boolean;
}

// ============================================================================
// Main Classes
// ============================================================================

/**
 * Represents an open PDF document for reading/analyzing
 *
 * @example
 * ```javascript
 * const { PdfDocument } = require('pdf_oxide');
 *
 * const doc = PdfDocument.open('document.pdf');
 * const pageCount = doc.pageCount;
 * const text = doc.extractText(0);
 * doc.close();
 * ```
 */
export class PdfDocument {
  /**
   * Gets the PDF version (e.g., "1.4", "1.7")
   * Getter property equivalent to getVersion()
   */
  readonly version: [number, number];

  /**
   * Gets the number of pages in the document
   * Getter property equivalent to getPageCount()
   */
  readonly pageCount: number;

  /**
   * Gets whether the document has a structure tree (tagged PDF)
   * Getter property equivalent to hasStructureTree()
   */
  readonly hasStructureTree: boolean;

  /**
   * Gets whether the document is encrypted
   */
  readonly isEncrypted: boolean;

  /**
   * Gets document metadata and information
   * Getter property equivalent to getDocumentInfo()
   * Cached for performance
   */
  readonly documentInfo: DocumentInfo;

  /**
   * Gets document XMP metadata
   * Getter property equivalent to getMetadata()
   * Cached for performance
   */
  readonly metadata: any;

  /**
   * Gets document forms (AcroForm)
   * Getter property equivalent to getForms()
   * Cached for performance
   */
  readonly forms: any | null;

  /**
   * Gets page labels for all pages
   * Getter property equivalent to getPageLabels()
   * Cached for performance
   */
  readonly pageLabels: PageLabel[];

  /**
   * Gets embedded files in the document
   * Getter property equivalent to getEmbeddedFiles()
   * Cached for performance
   */
  readonly embeddedFiles: EmbeddedFile[];

  /**
   * Opens a PDF document from a file path
   * @param filePath - Path to the PDF file
   * @returns Open PdfDocument instance
   * @throws {PdfIoError} If file cannot be opened
   * @throws {PdfParseError} If PDF is invalid
   */
  static open(filePath: string): PdfDocument;

  /**
   * Opens a PDF document with a password
   * @param filePath - Path to the PDF file
   * @param password - User or owner password
   * @returns Open PdfDocument instance
   * @throws {PdfEncryptionError} If password is incorrect
   */
  static openWithPassword(filePath: string, password: string): PdfDocument;

  /**
   * Gets a specific page (0-based indexing)
   * @param pageIndex - The page index (0-based)
   * @returns The requested page
   * @throws {PdfError} If page index is out of range
   */
  getPage(pageIndex: number): PdfPage;

  /**
   * Extracts all text from a specific page
   * @param pageIndex - The page index (0-based)
   * @returns Extracted text
   */
  extractText(pageIndex: number): string;

  /**
   * Asynchronously extracts text from a specific page
   * Runs on tokio thread pool to avoid blocking the event loop
   * @param pageIndex - The page index (0-based)
   * @returns Promise that resolves to extracted text
   */
  extractTextAsync(pageIndex: number): Promise<string>;

  /**
   * Extracts all text from all pages
   * @returns Complete document text
   */
  extractAllText(): string;

  /**
   * Converts page to Markdown format
   * @param pageIndex - The page index (0-based)
   * @param options - Conversion options (optional)
   * @returns Markdown formatted text
   */
  toMarkdown(pageIndex: number, options?: ConversionOptions): string;

  /**
   * Asynchronously converts page to Markdown format
   * Runs on tokio thread pool to avoid blocking the event loop
   * @param pageIndex - The page index (0-based)
   * @param options - Conversion options (optional)
   * @returns Promise that resolves to Markdown formatted text
   */
  toMarkdownAsync(pageIndex: number, options?: ConversionOptions): Promise<string>;

  /**
   * Converts all pages to Markdown
   * @param options - Conversion options (optional)
   * @returns Complete Markdown content
   */
  toMarkdownAll(options?: ConversionOptions): string;

  /**
   * Converts page to HTML format
   * @param pageIndex - The page index (0-based)
   * @param options - Conversion options (optional)
   * @returns HTML formatted text
   */
  toHtml(pageIndex: number, options?: ConversionOptions): string;

  /**
   * Converts all pages to HTML
   * @param options - Conversion options (optional)
   * @returns Complete HTML content
   */
  toHtmlAll(options?: ConversionOptions): string;

  /**
   * Searches for text in a specific page
   * @param text - Text to search for
   * @param pageIndex - The page index (0-based)
   * @param options - Search options (optional)
   * @returns Array of search results
   */
  search(text: string, pageIndex: number, options?: SearchOptions): SearchResult[];

  /**
   * Searches for text in all pages
   * @param text - Text to search for
   * @param options - Search options (optional)
   * @returns Array of search results from all pages
   */
  searchAll(text: string, options?: SearchOptions): SearchResult[];

  /**
   * Gets document metadata
   * @returns Document info object
   */
  getDocumentInfo(): DocumentInfo;

  /**
   * Validates the PDF structure
   * @returns Validation result with errors if any
   */
  validate(): ValidationResult;

  /**
   * Validate PDF/A conformance.
   * @param level - "1a"|"1b"|"2a"|"2b"|"2u"|"3a"|"3b"|"3u" (default "2b")
   * @returns Validation result with compliant flag, errors, and warnings
   */
  validatePdfA(level?: '1a' | '1b' | '2a' | '2b' | '2u' | '3a' | '3b' | '3u'): {
    compliant: boolean;
    errors: string[];
    warnings: string[];
  };

  /**
   * Convert document to PDF/A conformance in-place.
   * @param level - PDF/A level (default "2b")
   * @returns true on success
   */
  convertToPdfA(level?: '1a' | '1b' | '2a' | '2b' | '2u' | '3a' | '3b' | '3u'): boolean;

  /**
   * Validate PDF/X conformance.
   * @param level - "1a_2001"|"1a_2003"|"3_2003"|"4"|"5"|"6" (default "4")
   * @returns Validation result with compliant flag and errors
   */
  validatePdfX(level?: '1a_2001' | '1a_2003' | '3_2003' | '4' | '5' | '6'): {
    compliant: boolean;
    errors: string[];
    warnings: string[];
  };

  /**
   * Validate PDF/UA accessibility conformance.
   * @param level - "ua1" (default, only currently supported level)
   * @returns Accessibility result with accessible flag, errors, warnings, and element stats
   */
  validatePdfUA(level?: 'ua1'): {
    accessible: boolean;
    errors: string[];
    warnings: string[];
    stats: {
      structureElements: number;
      images: number;
      tables: number;
      formFields: number;
      annotations: number;
      pages: number;
    };
  };

  /**
   * Closes the document and releases resources
   */
  close(): void;
}

/**
 * Represents a document page
 */
export class PdfPage {
  /**
   * Gets the page width in points
   * Getter property equivalent to getWidth()
   */
  readonly width: number;

  /**
   * Gets the page height in points
   * Getter property equivalent to getHeight()
   */
  readonly height: number;

  /**
   * Gets the page index (0-based)
   * Getter property equivalent to getPageIndex()
   */
  readonly pageIndex: number;

  /**
   * Gets the page bounds (rectangle with position and dimensions)
   * Getter property that calculates bounds from position and dimensions
   */
  readonly bounds: Rect;

  /**
   * Gets the page orientation ('portrait' or 'landscape')
   * Computed property based on width and height
   */
  readonly orientation: 'portrait' | 'landscape';

  /**
   * Gets the page aspect ratio (width / height)
   * Computed property for easy layout calculations
   */
  readonly aspectRatio: number;

  /**
   * Gets all elements on this page
   * @returns Array of PDF elements
   */
  getElements(): PdfElement[];

  /**
   * Gets text elements on this page
   * @returns Array of text elements
   */
  getTextElements(): PdfText[];

  /**
   * Gets image elements on this page
   * @returns Array of image elements
   */
  getImageElements(): PdfImage[];
}

/**
 * Base class for PDF elements
 */
export class PdfElement {
  /**
   * Gets the bounding box of this element
   */
  readonly bounds: Rect;

  /**
   * Gets the element type
   */
  readonly type: string;
}

/**
 * Represents a text element
 */
export class PdfText extends PdfElement {
  /**
   * Gets the text content
   */
  readonly text: string;

  /**
   * Gets the font name
   */
  readonly fontName: string;

  /**
   * Gets the font size
   */
  readonly fontSize: number;

  /**
   * Gets the text color
   */
  readonly color: Color;

  /**
   * Gets whether text is bold
   */
  readonly isBold: boolean;

  /**
   * Gets whether text is italic
   */
  readonly isItalic: boolean;
}

/**
 * Represents an image element
 */
export class PdfImage extends PdfElement {
  /**
   * Gets the image width in pixels
   */
  readonly width: number;

  /**
   * Gets the image height in pixels
   */
  readonly height: number;

  /**
   * Gets the image format (JPEG, PNG, etc.)
   */
  readonly format: string;

  /**
   * Gets the image horizontal DPI
   */
  readonly horizontalDpi: number;

  /**
   * Gets the image vertical DPI
   */
  readonly verticalDpi: number;

  /**
   * Gets whether image is grayscale
   */
  readonly isGrayscale: boolean;

  /**
   * Gets the image data as Buffer
   */
  getData(): Buffer;
}

/**
 * Represents a path element (vector graphics)
 */
export class PdfPath extends PdfElement {
  /**
   * Gets the stroke color (if any)
   */
  readonly strokeColor?: Color;

  /**
   * Gets the fill color (if any)
   */
  readonly fillColor?: Color;

  /**
   * Gets the line width
   */
  readonly lineWidth: number;
}

/**
 * Represents a table element
 */
export class PdfTable extends PdfElement {
  /**
   * Gets the number of rows
   */
  readonly rowCount: number;

  /**
   * Gets the number of columns
   */
  readonly columnCount: number;

  /**
   * Gets a specific row
   * @param rowIndex - The row index
   * @returns Array of cell contents
   */
  getRow(rowIndex: number): string[];

  /**
   * Gets a specific column
   * @param columnIndex - The column index
   * @returns Array of cell contents
   */
  getColumn(columnIndex: number): string[];
}

/**
 * Represents a structure element (tagged PDF)
 */
export class PdfStructure extends PdfElement {
  /**
   * Gets the structure type
   */
  readonly structureType: string;

  /**
   * Gets the alternative text (accessibility)
   */
  readonly altText?: string;

  /**
   * Gets the actual text content
   */
  readonly actualText?: string;

  /**
   * Gets child structure elements
   */
  getChildren(): PdfStructure[];
}

/**
 * Represents an annotation
 */
export class Annotation {
  /**
   * Gets the annotation type
   */
  readonly type: string;

  /**
   * Gets the author
   */
  readonly author?: string;

  /**
   * Gets the creation date
   */
  readonly createdAt?: Date;

  /**
   * Gets the modification date
   */
  readonly modifiedAt?: Date;
}

/**
 * Represents a text annotation (comment, sticky note, etc.)
 */
export class TextAnnotation extends Annotation {
  /**
   * Gets the annotation content
   */
  readonly content: string;

  /**
   * Gets the annotation subject
   */
  readonly subject?: string;

  /**
   * Gets the icon name (Note, Help, etc.)
   */
  readonly icon?: string;
}

/**
 * Represents a highlight annotation
 */
export class HighlightAnnotation extends Annotation {
  /**
   * Gets the highlighted color
   */
  readonly color: Color;

  /**
   * Gets the bounding box of the highlight
   */
  readonly bounds: Rect;
}

/**
 * Represents a link annotation
 */
export class LinkAnnotation extends Annotation {
  /**
   * Gets the link URL
   */
  readonly url?: string;

  /**
   * Gets the target page (if internal link)
   */
  readonly targetPage?: number;

  /**
   * Gets the bounding box of the link
   */
  readonly bounds: Rect;
}

// ============================================================================
// PDF Creation Classes
// ============================================================================

/**
 * Represents a PDF document being created/edited
 *
 * @example
 * ```javascript
 * const { Pdf, Rect, Color } = require('pdf_oxide');
 *
 * const pdf = Pdf.create();
 * pdf.addPage(600, 800);
 * pdf.addText("Hello World", 50, 50);
 * pdf.save('output.pdf');
 * ```
 */
export class Pdf {
  /**
   * Gets the document title
   */
  title?: string;

  /**
   * Gets the document author
   */
  author?: string;

  /**
   * Gets the document subject
   */
  subject?: string;

  /**
   * Gets the document keywords
   */
  keywords?: string[];

  /**
   * Gets the number of pages
   * Getter property equivalent to getPageCount()
   */
  readonly pageCount: number;

  /**
   * Gets the PDF version (e.g., "1.4", "1.7")
   * Getter property equivalent to getVersion()
   */
  readonly version: [number, number];

  /**
   * Gets document metadata and information
   * Getter property equivalent to getDocumentInfo()
   * Cached for performance
   */
  readonly documentInfo: DocumentInfo;

  /**
   * Gets document XMP metadata
   * Getter property equivalent to getMetadata()
   * Cached for performance
   */
  readonly metadata: any;

  /**
   * Gets document forms (AcroForm)
   * Getter property equivalent to getForms()
   * Cached for performance
   */
  readonly forms: any | null;

  /**
   * Creates a new empty PDF document
   */
  static create(): Pdf;

  /**
   * Creates a PDF from Markdown content
   * @param markdown - Markdown formatted content
   * @returns New Pdf instance
   */
  static fromMarkdown(markdown: string): Pdf;

  /**
   * Creates a PDF from HTML content
   * @param html - HTML formatted content
   * @returns New Pdf instance
   */
  static fromHtml(html: string): Pdf;

  /**
   * Creates a PDF from plain text
   * @param text - Plain text content
   * @returns New Pdf instance
   */
  static fromText(text: string): Pdf;

  /**
   * Creates a PDF from HTML with CSS styling and a single embedded font.
   * @param html - HTML content string
   * @param css - CSS stylesheet string
   * @param fontBytes - Font file bytes (TTF/OTF) used for body text
   * @returns New Pdf instance
   */
  static fromHtmlCss(html: string, css: string, fontBytes: Buffer | Uint8Array): Pdf;

  /**
   * Creates a PDF from HTML with CSS styling and a multi-font cascade.
   * `families` and `fonts` are parallel arrays: `families[i]` is the
   * CSS `font-family` name that resolves to `fonts[i]`.
   * @param html - HTML content string
   * @param css - CSS stylesheet string
   * @param families - Font family names (parallel to `fonts`)
   * @param fonts - Font file bytes arrays (parallel to `families`)
   * @returns New Pdf instance
   */
  static fromHtmlCssWithFonts(
    html: string,
    css: string,
    families: string[],
    fonts: (Buffer | Uint8Array)[]
  ): Pdf;

  /**
   * Adds a new page to the document
   * @param width - Page width in points
   * @param height - Page height in points
   * @returns The new page index
   */
  addPage(width: number, height: number): number;

  /**
   * Adds a page with standard size
   * @param pageSize - Standard page size
   * @returns The new page index
   */
  addPageWithSize(pageSize: PageSize): number;

  /**
   * Adds text to the current page
   * @param text - Text to add
   * @param x - X coordinate
   * @param y - Y coordinate
   * @param fontSize - Font size (default: 12)
   * @param color - Text color (default: black)
   */
  addText(text: string, x: number, y: number, fontSize?: number, color?: Color): void;

  /**
   * Adds an image from file
   * @param filePath - Path to image file
   * @param x - X coordinate
   * @param y - Y coordinate
   * @param width - Image width
   * @param height - Image height
   */
  addImage(filePath: string, x: number, y: number, width: number, height: number): void;

  /**
   * Adds a rectangle shape
   * @param x - X coordinate
   * @param y - Y coordinate
   * @param width - Width
   * @param height - Height
   * @param color - Fill color (optional)
   * @param strokeColor - Stroke color (optional)
   * @param strokeWidth - Stroke width (optional)
   */
  addRect(
    x: number,
    y: number,
    width: number,
    height: number,
    color?: Color,
    strokeColor?: Color,
    strokeWidth?: number
  ): void;

  /**
   * Adds a link annotation
   * @param bounds - Link bounds
   * @param url - URL to link to
   */
  addLink(bounds: Rect, url: string): void;

  /**
   * Adds a comment annotation
   * @param x - X coordinate
   * @param y - Y coordinate
   * @param content - Comment text
   * @param author - Author name (optional)
   */
  addComment(x: number, y: number, content: string, author?: string): void;

  /**
   * Saves the PDF to a file
   * @param filePath - Path where to save
   */
  save(filePath: string): void;

  /**
   * Asynchronously saves the PDF to a file
   * Runs on tokio thread pool to avoid blocking the event loop
   * @param filePath - Path where to save
   * @returns Promise that resolves when save is complete
   */
  saveAsync(filePath: string): Promise<void>;

  /**
   * Saves the PDF and returns the buffer
   * @returns PDF content as Buffer
   */
  saveToBuffer(): Buffer;
}

/**
 * Builder for creating PDF documents with fluent API
 *
 * @example
 * ```javascript
 * const { PdfBuilder } = require('pdf_oxide');
 *
 * const pdf = new PdfBuilder()
 *   .withTitle('Report')
 *   .withAuthor('System')
 *   .addPage(600, 800)
 *   .addText('Hello World', 50, 50)
 *   .save('output.pdf');
 * ```
 */
export class PdfBuilder {
  /**
   * Sets the document title
   */
  withTitle(title: string): PdfBuilder;

  /**
   * Sets the document author
   */
  withAuthor(author: string): PdfBuilder;

  /**
   * Sets the document subject
   */
  withSubject(subject: string): PdfBuilder;

  /**
   * Sets the document keywords
   */
  withKeywords(keywords: string[]): PdfBuilder;

  /**
   * Adds a page
   */
  addPage(width: number, height: number): PdfBuilder;

  /**
   * Adds a page with standard size
   */
  addPageWithSize(pageSize: PageSize): PdfBuilder;

  /**
   * Adds text to the current page
   */
  addText(text: string, x: number, y: number, fontSize?: number, color?: Color): PdfBuilder;

  /**
   * Adds an image from file
   */
  addImage(filePath: string, x: number, y: number, width: number, height: number): PdfBuilder;

  /**
   * Builds and returns the PDF
   */
  build(): Pdf;

  /**
   * Builds and saves the PDF to a file
   */
  save(filePath: string): void;
}

/**
 * Text search utility for efficient searching
 */
export class TextSearcher {
  /**
   * Creates a text searcher for a document
   * @param document - The PDF document to search in
   */
  constructor(document: PdfDocument);

  /**
   * Searches for text on a specific page
   * @param text - Text to search for
   * @param pageIndex - The page index
   * @param options - Search options
   */
  search(text: string, pageIndex: number, options?: SearchOptions): SearchResult[];

  /**
   * Searches for text across all pages
   * @param text - Text to search for
   * @param options - Search options
   */
  searchAll(text: string, options?: SearchOptions): SearchResult[];
}

// ============================================================================
// Supporting Interfaces
// ============================================================================

/**
 * Document metadata information
 */
export interface DocumentInfo {
  title?: string;
  author?: string;
  subject?: string;
  keywords?: string[];
  creator?: string;
  producer?: string;
  createdAt?: Date;
  modifiedAt?: Date;
}

/**
 * Result of PDF validation
 */
export interface ValidationResult {
  /**
   * Whether the PDF is valid
   */
  isValid: boolean;

  /**
   * Array of validation errors (if any)
   */
  errors?: ValidationError[];

  /**
   * Array of validation warnings (if any)
   */
  warnings?: string[];
}

/**
 * A validation error in the PDF
 */
export interface ValidationError {
  /**
   * The error type
   */
  type: string;

  /**
   * The error message
   */
  message: string;

  /**
   * The page where error occurred (if applicable)
   */
  pageIndex?: number;
}

// ============================================================================
// Builders - Fluent APIs for Configuration
// ============================================================================

/**
 * Fluent builder for creating PDF documents with metadata
 *
 * @example
 * ```typescript
 * const pdf = PdfBuilder.create()
 *   .title('My Document')
 *   .author('John Doe')
 *   .fromMarkdown('# Content');
 * ```
 */
export class PdfBuilder {
  static create(): PdfBuilder;

  title(title: string): PdfBuilder;
  author(author: string): PdfBuilder;
  subject(subject: string): PdfBuilder;
  keywords(keywords: string[]): PdfBuilder;
  addKeyword(keyword: string): PdfBuilder;
  pageSize(size: string): PdfBuilder;
  margins(top: number, right: number, bottom: number, left: number): PdfBuilder;

  fromMarkdown(markdown: string): Pdf;
  fromHtml(html: string): Pdf;
  fromText(text: string): Pdf;
  fromHtmlCss(html: string, css: string, fontBytes: Buffer | Uint8Array): Pdf;
  fromHtmlCssWithFonts(
    html: string,
    css: string,
    families: string[],
    fonts: (Buffer | Uint8Array)[]
  ): Pdf;

  fromMarkdownAsync(markdown: string): Promise<Pdf>;
  fromHtmlAsync(html: string): Promise<Pdf>;
  fromTextAsync(text: string): Promise<Pdf>;
}

/**
 * Fluent builder for PDF conversion options
 *
 * @example
 * ```typescript
 * const options = ConversionOptionsBuilder.create()
 *   .preserveFormatting(true)
 *   .detectTables(true)
 *   .build();
 * ```
 */
export class ConversionOptionsBuilder {
  static create(): ConversionOptionsBuilder;
  static default(): ConversionOptions;
  static textOnly(): ConversionOptions;
  static highQuality(): ConversionOptions;
  static fast(): ConversionOptions;

  preserveFormatting(preserve: boolean): ConversionOptionsBuilder;
  detectHeadings(detect: boolean): ConversionOptionsBuilder;
  detectTables(detect: boolean): ConversionOptionsBuilder;
  detectLists(detect: boolean): ConversionOptionsBuilder;
  includeImages(include: boolean): ConversionOptionsBuilder;
  imageFormat(format: string): ConversionOptionsBuilder;
  imageQuality(quality: number): ConversionOptionsBuilder;
  maxImageDimension(max: number): ConversionOptionsBuilder;
  outputEncoding(encoding: string): ConversionOptionsBuilder;
  normalizeWhitespace(normalize: boolean): ConversionOptionsBuilder;
  extractAnnotations(extract: boolean): ConversionOptionsBuilder;
  useStructureTree(use: boolean): ConversionOptionsBuilder;
  pageRange(start: number, end: number): ConversionOptionsBuilder;

  build(): ConversionOptions;
}

/**
 * Fluent builder for document metadata
 *
 * @example
 * ```typescript
 * const metadata = MetadataBuilder.create()
 *   .title('Document')
 *   .author('Author')
 *   .build();
 * ```
 */
export class MetadataBuilder {
  static create(): MetadataBuilder;

  title(title: string): MetadataBuilder;
  author(author: string): MetadataBuilder;
  subject(subject: string): MetadataBuilder;
  keywords(keywords: string[]): MetadataBuilder;
  addKeyword(keyword: string): MetadataBuilder;
  creator(creator: string): MetadataBuilder;
  producer(producer: string): MetadataBuilder;
  creationDate(date: Date): MetadataBuilder;
  modificationDate(date: Date): MetadataBuilder;
  customProperty(key: string, value: string): MetadataBuilder;
  customProperties(properties: Record<string, string>): MetadataBuilder;
  withCurrentDate(): MetadataBuilder;

  build(): Metadata;
}

/**
 * Fluent builder for PDF annotations
 *
 * @example
 * ```typescript
 * const annotation = AnnotationBuilder.create()
 *   .type('highlight')
 *   .content('Important')
 *   .colorName('yellow')
 *   .build();
 * ```
 */
export class AnnotationBuilder {
  static create(): AnnotationBuilder;

  type(type: string): AnnotationBuilder;
  asText(): AnnotationBuilder;
  asHighlight(): AnnotationBuilder;
  asUnderline(): AnnotationBuilder;
  asStrikeout(): AnnotationBuilder;
  asSquiggly(): AnnotationBuilder;

  content(content: string): AnnotationBuilder;
  author(author: string): AnnotationBuilder;
  subject(subject: string): AnnotationBuilder;
  color(rgb: [number, number, number]): AnnotationBuilder;
  colorName(name: string): AnnotationBuilder;
  opacity(opacity: number): AnnotationBuilder;
  bounds(bounds: { x: number; y: number; width: number; height: number }): AnnotationBuilder;
  creationDate(date: Date): AnnotationBuilder;
  modificationDate(date: Date): AnnotationBuilder;

  printable(): AnnotationBuilder;
  notPrintable(): AnnotationBuilder;
  locked(locked: boolean): AnnotationBuilder;
  reply(content: string): AnnotationBuilder;

  build(): Annotation;
}

/**
 * Fluent builder for search options
 *
 * @example
 * ```typescript
 * const options = SearchOptionsBuilder.create()
 *   .caseSensitive(false)
 *   .wholeWords(true)
 *   .build();
 * ```
 */
export class SearchOptionsBuilder {
  static create(): SearchOptionsBuilder;
  static default(): SearchOptions;
  static strict(): SearchOptions;
  static regex(): SearchOptions;

  caseSensitive(sensitive: boolean): SearchOptionsBuilder;
  wholeWords(whole: boolean): SearchOptionsBuilder;
  useRegex(regex: boolean): SearchOptionsBuilder;
  ignoreAccents(ignore: boolean): SearchOptionsBuilder;
  maxResults(max: number): SearchOptionsBuilder;
  searchAnnotations(search: boolean): SearchOptionsBuilder;

  build(): SearchOptions;
}

/**
 * Metadata object returned by MetadataBuilder
 */
export interface Metadata {
  title?: string;
  author?: string;
  subject?: string;
  keywords: string[];
  creator?: string;
  producer: string;
  creationDate: Date;
  modificationDate: Date;
  customProperties: Record<string, string>;
}

/**
 * Annotation object returned by AnnotationBuilder
 */
export interface Annotation {
  type: string;
  content: string;
  author?: string;
  subject?: string;
  color: [number, number, number];
  opacity: number;
  bounds?: { x: number; y: number; width: number; height: number };
  creationDate: Date;
  modificationDate: Date;
  flags: number;
  reply?: string;
}

// ============================================================================
// Manager Classes - Domain-specific Operations
// ============================================================================

/**
 * Manager for PDF document outlines (bookmarks)
 */
export class OutlineManager {
  constructor(document: PdfDocument);
  hasOutlines(): boolean;
  getOutlineCount(): number;
  getOutlines(): Array<any>;
  findByTitle(titleFragment: string): any | null;
  findAllByTitle(titleFragment: string): Array<any>;
  getOutlinesForPage(pageIndex: number): Array<any>;
  pageHasOutlines(pageIndex: number): boolean;
  getOutlineAt(index: number): any | null;
  containsPageNumber(pageNumber: number): boolean;
}

/**
 * Manager for document metadata
 */
export class MetadataManager {
  constructor(document: PdfDocument);
  getTitle(): string | null;
  getAuthor(): string | null;
  getSubject(): string | null;
  getKeywords(): string[];
  getCreator(): string | null;
  getProducer(): string | null;
  getCreationDate(): Date | null;
  getModificationDate(): Date | null;
  getAllMetadata(): Record<string, any>;
  hasMetadata(): boolean;
  getMetadataSummary(): string;
  hasKeyword(keyword: string): boolean;
  getKeywordCount(): number;
  compareWith(otherDocument: PdfDocument): {
    matching: Record<string, any>;
    differing: Record<string, any>;
  };
  validate(): { isComplete: boolean; issues: string[]; missingFieldCount: number };
}

/**
 * Manager for content extraction from PDFs
 */
export class ExtractionManager {
  constructor(document: PdfDocument);
  extractText(pageIndex: number, options?: ConversionOptions): string;
  extractAllText(options?: ConversionOptions): string;
  extractTextRange(
    startPageIndex: number,
    endPageIndex: number,
    options?: ConversionOptions
  ): string;
  extractMarkdown(pageIndex: number, options?: ConversionOptions): string;
  extractAllMarkdown(options?: ConversionOptions): string;
  extractMarkdownRange(
    startPageIndex: number,
    endPageIndex: number,
    options?: ConversionOptions
  ): string;
  getPageWordCount(pageIndex: number): number;
  getTotalWordCount(): number;
  getPageCharacterCount(pageIndex: number): number;
  getTotalCharacterCount(): number;
  getPageLineCount(pageIndex: number): number;
  getContentStatistics(): {
    pageCount: number;
    wordCount: number;
    characterCount: number;
    averageWordsPerPage: number;
    averageCharactersPerPage: number;
  };
  searchContent(
    searchText: string,
    contextLength?: number
  ): Array<{
    pageIndex: number;
    pageNumber: number;
    matchIndex: number;
    snippet: string;
    matchText: string;
  }>;
}

/**
 * Manager for text search operations
 */
export class SearchManager {
  constructor(document: PdfDocument);
  search(searchText: string, pageIndex: number, options?: SearchOptions): SearchResult[];
  searchAll(searchText: string, options?: SearchOptions): SearchResult[];
  countOccurrences(searchText: string, pageIndex: number, options?: SearchOptions): number;
  countAllOccurrences(searchText: string, options?: SearchOptions): number;
  contains(searchText: string, pageIndex: number, options?: SearchOptions): boolean;
  containsAnywhere(searchText: string, options?: SearchOptions): boolean;
  getPagesContaining(searchText: string, options?: SearchOptions): number[];
  getSearchStatistics(
    searchText: string,
    options?: SearchOptions
  ): {
    searchText: string;
    totalOccurrences: number;
    pagesContaining: number;
    firstMatchPage: number;
    lastMatchPage: number;
    pages: number[];
    occurrencesPerPage: Array<any>;
  };
  searchRegex(pattern: RegExp | string, options?: SearchOptions): SearchResult[];
  findFirst(searchText: string, options?: SearchOptions): SearchResult | null;
  findLast(searchText: string, options?: SearchOptions): SearchResult | null;
  highlightMatches(searchText: string, options?: SearchOptions): SearchResult[];
  isSearchable(): boolean;
  getCapabilities(): {
    caseSensitiveSearch: boolean;
    wholeWordSearch: boolean;
    regexSearch: boolean;
    annotationSearch: boolean;
    maxResults: number;
    isSearchable: boolean;
  };
}

/**
 * Manager for PDF document security
 */
export class SecurityManager {
  constructor(document: PdfDocument);
  isEncrypted(): boolean;
  requiresPassword(): boolean;
  getEncryptionAlgorithm(): string | null;
  canPrint(): boolean;
  canCopy(): boolean;
  canModify(): boolean;
  canAnnotate(): boolean;
  canFillForms(): boolean;
  isViewOnly(): boolean;
  getPermissionsSummary(): {
    canPrint: boolean;
    canCopy: boolean;
    canModify: boolean;
    canAnnotate: boolean;
    canFillForms: boolean;
    isViewOnly: boolean;
    isEncrypted: boolean;
    requiresPassword: boolean;
    encryptionAlgorithm: string | null;
  };
  getSecurityLevel(): {
    level: string;
    description: string;
    isEncrypted: boolean;
    algorithm: string | null;
    restrictedAccess: boolean;
  };
  validateAccessibility(): {
    canExtractText: boolean;
    canExtractImages: boolean;
    canAnalyzeContent: boolean;
    canSearch: boolean;
    canViewContent: boolean;
    isAccessible: boolean;
    issues: string[];
  };
  generateSecurityReport(): string;
}

/**
 * Manager for page annotations
 */
export class AnnotationManager {
  constructor(page: PdfPage);
  getAnnotations(): Annotation[];
  getAnnotationsByType(type: string): Annotation[];
  getAnnotationCount(): number;
  getAnnotationsByAuthor(author: string): Annotation[];
  getAnnotationAuthors(): string[];
  getAnnotationsAfter(date: Date): Annotation[];
  getAnnotationsBefore(date: Date): Annotation[];
  getAnnotationsWithContent(contentFragment: string): Annotation[];
  getHighlights(): Annotation[];
  getComments(): Annotation[];
  getUnderlines(): Annotation[];
  getStrikeouts(): Annotation[];
  getSquigglies(): Annotation[];
  getAnnotationStatistics(): {
    total: number;
    byType: Record<string, number>;
    byAuthor: Record<string, number>;
    authors: string[];
    types: string[];
    hasComments: boolean;
    hasHighlights: boolean;
    averageOpacity: number;
    recentModifications: number;
  };
  getRecentAnnotations(days: number): Annotation[];
  generateAnnotationSummary(): string;
  validateAnnotation(annotation: Annotation): { isValid: boolean; issues: string[] };
}

/**
 * Options for rendering pages to images
 */
export class RenderOptions {
  dpi: number;
  format: 'png' | 'jpeg';
  quality: number;
  maxWidth: number | null;
  maxHeight: number | null;

  constructor(config?: {
    dpi?: number;
    format?: 'png' | 'jpeg';
    quality?: number;
    maxWidth?: number;
    maxHeight?: number;
  });

  static merge(options: RenderOptions | object | null | undefined): RenderOptions;
  static fromQuality(quality: 'draft' | 'normal' | 'high'): RenderOptions;
  toJSON(): {
    dpi: number;
    format: 'png' | 'jpeg';
    quality: number;
    maxWidth: number | null;
    maxHeight: number | null;
  };
}

/**
 * Manager for PDF page rendering and image output
 */
export class RenderingManager {
  constructor(document: PdfDocument);

  // Existing methods
  clearCache(): void;
  getMaxResolution(): number;
  getSupportedColorSpaces(): string[];
  getPageDimensions(pageIndex: number): { width: number; height: number; unit: string };
  getDisplaySize(
    pageIndex: number,
    zoomLevel: number
  ): { width: number; height: number; unit: string };
  getPageRotation(pageIndex: number): number;
  getPageCropBox(pageIndex: number): { x: number; y: number; width: number; height: number };
  getPageMediaBox(pageIndex: number): { x: number; y: number; width: number; height: number };
  getPageBleedBox(pageIndex: number): { x: number; y: number; width: number; height: number };
  getPageTrimBox(pageIndex: number): { x: number; y: number; width: number; height: number };
  getPageArtBox(pageIndex: number): { x: number; y: number; width: number; height: number };
  calculateZoomForWidth(pageIndex: number, viewportWidth: number): number;
  calculateZoomForHeight(pageIndex: number, viewportHeight: number): number;
  calculateZoomToFit(pageIndex: number, viewportWidth: number, viewportHeight: number): number;
  getEmbeddedFonts(pageIndex: number): Array<{ name: string; embedded: boolean; subset?: boolean }>;
  getEmbeddedImages(
    pageIndex: number
  ): Array<{ name: string; width: number; height: number; colorSpace?: string }>;
  getPageResources(pageIndex: number): {
    fonts: Array<{ name: string; embedded: boolean; subset?: boolean }>;
    images: Array<{ name: string; width: number; height: number; colorSpace?: string }>;
    colorSpaces: string[];
    patterns: any[];
  };
  getRecommendedResolution(quality: 'draft' | 'normal' | 'high'): number;
  getRenderingStatistics(): {
    totalFonts: number;
    totalImages: number;
    avgPageSize: number;
    colorSpaceCount: number;
    pageCount: number;
    maxResolution: number;
  };
  canRenderPage(pageIndex: number): boolean;
  validateRenderingState(): { isValid: boolean; issues: string[] };

  // New rendering methods
  renderPageToFile(
    pageIndex: number,
    outputPath: string,
    options?: RenderOptions | object | null
  ): Promise<string>;
  renderPageToBytes(pageIndex: number, options?: RenderOptions | object | null): Promise<Buffer>;
  renderPagesRange(
    startPage: number,
    endPage: number,
    outputDir: string,
    namePattern?: string,
    options?: RenderOptions | object | null
  ): Promise<string[]>;
}

// ============================================================================
// Module Exports Summary
// ============================================================================

/**
 * PDF Oxide Module - Complete PDF toolkit
 *
 * ## Version Functions
 * - getVersion(): string
 * - getPdfOxideVersion(): string
 *
 * ## Main Classes
 * - PdfDocument (read, analyze, search)
 * - Pdf (create, edit)
 * - PdfPage
 *
 * ## Builders (Fluent Configuration)
 * - PdfBuilder (document creation with metadata)
 * - ConversionOptionsBuilder (PDF format conversion)
 * - MetadataBuilder (document metadata)
 * - AnnotationBuilder (PDF annotations)
 * - SearchOptionsBuilder (text search configuration)
 *
 * ## Managers (Domain-specific Operations)
 * - OutlineManager (document bookmarks)
 * - MetadataManager (document metadata)
 * - ExtractionManager (content extraction)
 * - SearchManager (text search)
 * - SecurityManager (encryption & permissions)
 * - AnnotationManager (page annotations)
 *
 * ## Element Types
 * - PdfElement (base)
 * - PdfText
 * - PdfImage
 * - PdfPath
 * - PdfTable
 * - PdfStructure
 *
 * ## Annotations
 * - Annotation (base)
 * - TextAnnotation
 * - HighlightAnnotation
 * - LinkAnnotation
 *
 * ## Error Classes
 * - PdfError
 * - PdfIoError
 * - PdfParseError
 * - PdfEncryptionError
 * - PdfUnsupportedError
 * - PdfInvalidStateError
 * - PdfDecodeError
 * - PdfEncodeError
 * - PdfFontError
 * - PdfImageError
 * - PdfCircularReferenceError
 * - PdfRecursionLimitError
 *
 * ## Type Classes
 * - Rect
 * - Point
 * - Color
 * - PageSize (enum)
 * - SearchOptions
 * - SearchResult
 * - ConversionOptions
 *
 * ## Utilities
 * - TextSearcher
 *
 * @example
 * ```typescript
 * import {
 *   PdfDocument,
 *   Pdf,
 *   PdfBuilder,
 *   SearchOptions,
 *   PageSize,
 *   Color,
 *   PdfError
 * } from 'pdf_oxide';
 *
 * // Read and analyze
 * const doc = PdfDocument.open('input.pdf');
 * const text = doc.extractText(0);
 * const results = doc.searchAll('keyword');
 * doc.close();
 *
 * // Create with builder
 * const pdf = new PdfBuilder()
 *   .withTitle('My PDF')
 *   .addPageWithSize(PageSize.A4)
 *   .addText('Hello World', 50, 50, 12, new Color(0, 0, 0))
 *   .save('output.pdf');
 * ```
 */

// ============================================================================
// DocumentBuilder — v0.3.39 table primitives (#393)
// ============================================================================

/**
 * Horizontal alignment for `textInRect` and table cells.
 * Integer values match the C FFI encoding.
 */
export enum Align {
  Left = 0,
  Center = 1,
  Right = 2,
}

/** Column descriptor for {@link TableSpec} / {@link StreamingTableConfig}. */
export interface Column {
  /** Header label (bold when `hasHeader` / `repeatHeader` is true). */
  header: string;
  /** Column width in PDF points. */
  width: number;
  /** Cell alignment (default Align.Left). */
  align?: Align;
}

/** Buffered-table spec consumed by `PageBuilder.table(...)`. */
export interface TableSpec {
  columns: Column[];
  rows: Array<Array<string | null | undefined>>;
  /** Promote columns to a styled header row. Default `true`. */
  hasHeader?: boolean;
}

/** Config passed to `PageBuilder.streamingTable(...)`. */
export interface StreamingTableConfig {
  columns: Column[];
  /** Emit the header on completion. Default `true`. */
  repeatHeader?: boolean;
}

/**
 * Managed streaming-table adapter. Rows pushed via `pushRow` /
 * `pushAll` are buffered and flushed through the buffered-table FFI
 * on `finish()`.
 */
export class StreamingTable {
  /** Push a single row (cell count must equal `columns.length`). */
  pushRow(cells: Array<string | null | undefined>): this;
  /** Drain a sync or async iterable of rows. */
  pushAll(
    rows:
      | Iterable<Array<string | null | undefined>>
      | AsyncIterable<Array<string | null | undefined>>
  ): Promise<this>;
  /** Flush buffered rows and return the parent PageBuilder. */
  finish(): Promise<PageBuilder>;
  /** Number of body rows buffered so far. */
  readonly rowCount: number;
}

/**
 * Fluent document builder — the programmatic multi-page construction API
 * exposed through the C FFI.
 */
export class DocumentBuilder {
  static create(): DocumentBuilder;
  title(title: string): this;
  author(author: string): this;
  subject(subject: string): this;
  keywords(keywords: string): this;
  creator(creator: string): this;
  onOpen(script: string): this;
  /**
   * Enable PDF/UA-1 tagged PDF mode.
   * Emits /MarkInfo, /StructTreeRoot, /Lang, and /ViewerPreferences in the
   * catalog. Opt-in — no effect unless called. Bundle F-1/F-2.
   */
  taggedPdfUa1(): this;
  /**
   * Set the document's natural language tag, e.g. "en-US".
   * Emitted as /Lang in the catalog when taggedPdfUa1() is set. Bundle F-2.
   */
  language(lang: string): this;
  /**
   * Add a role-map entry: custom structure type → standard PDF structure type.
   * Emitted in /RoleMap inside the StructTreeRoot when taggedPdfUa1() is set.
   * Multiple calls accumulate entries. Bundle F-4.
   */
  roleMap(custom: string, standard: string): this;
  registerEmbeddedFont(name: string, font: EmbeddedFont): this;
  a4Page(): PageBuilder;
  letterPage(): PageBuilder;
  page(width: number, height: number): PageBuilder;
  build(): Buffer;
  save(path: string): void;
  saveEncrypted(path: string, userPassword: string, ownerPassword: string): void;
  toBytesEncrypted(userPassword: string, ownerPassword: string): Buffer;
  close(): void;
  [Symbol.dispose](): void;
}

/** TTF / OTF font registerable with {@link DocumentBuilder}. */
export class EmbeddedFont {
  static fromFile(path: string): EmbeddedFont;
  static fromBytes(data: Uint8Array | Buffer, name?: string): EmbeddedFont;
  close(): void;
  [Symbol.dispose](): void;
}

/**
 * Fluent per-page builder returned by `DocumentBuilder.a4Page()` etc.
 * Single-use — `done()` commits the page.
 */
export class PageBuilder {
  // --- Text / typography --------------------------------------------
  font(name: string, size: number): this;
  at(x: number, y: number): this;
  text(text: string): this;
  heading(level: number, text: string): this;
  paragraph(text: string): this;
  space(points: number): this;
  horizontalRule(): this;

  // --- Annotations ---------------------------------------------------
  linkUrl(url: string): this;
  linkPage(pageIndex: number): this;
  linkNamed(destination: string): this;
  linkJavascript(script: string): this;
  onOpen(script: string): this;
  onClose(script: string): this;
  fieldKeystroke(script: string): this;
  fieldFormat(script: string): this;
  fieldValidate(script: string): this;
  fieldCalculate(script: string): this;
  highlight(r: number, g: number, b: number): this;
  underline(r: number, g: number, b: number): this;
  strikeout(r: number, g: number, b: number): this;
  squiggly(r: number, g: number, b: number): this;
  stickyNote(text: string): this;
  stickyNoteAt(x: number, y: number, text: string): this;
  watermark(text: string): this;
  watermarkConfidential(): this;
  watermarkDraft(): this;
  stamp(typeName: string): this;
  freeText(x: number, y: number, w: number, h: number, text: string): this;

  // --- Form fields ---------------------------------------------------
  textField(name: string, x: number, y: number, w: number, h: number, defaultValue?: string): this;
  checkbox(name: string, x: number, y: number, w: number, h: number, checked?: boolean): this;
  comboBox(
    name: string,
    x: number,
    y: number,
    w: number,
    h: number,
    options: string[],
    selected?: string
  ): this;
  radioGroup(
    name: string,
    buttons: Array<[string, number, number, number, number]>,
    selected?: string
  ): this;
  pushButton(name: string, x: number, y: number, w: number, h: number, caption: string): this;
  /** Add an unsigned signature placeholder field (/FT /Sig) at the given bounds. */
  signatureField(name: string, x: number, y: number, w: number, h: number): this;
  /** Add a footnote: inline refMark at cursor + noteText near page bottom with separator. */
  footnote(refMark: string, noteText: string): this;
  /** Lay out text as balanced multi-column flow (columnCount columns, gapPt between them). Paragraphs separated by "\n\n". */
  columns(columnCount: number, gapPt: number, text: string): this;
  /** Emit text inline at the current horizontal cursor position (no line break). */
  inline(text: string): this;
  /** Emit text inline in bold weight. */
  inlineBold(text: string): this;
  /** Emit text inline in italic style. */
  inlineItalic(text: string): this;
  /** Emit text inline in an RGB colour (channels 0–1). */
  inlineColor(r: number, g: number, b: number, text: string): this;
  /** Advance the cursor to the start of the next line. */
  newline(): this;

  // --- Barcode / QR-code placement ----------------------------------
  /** Place a 1-D barcode image on the page. barcodeType: 0=Code128 1=Code39 2=EAN13 3=EAN8 4=UPCA 5=ITF 6=Code93 7=Codabar. */
  barcode1d(barcodeType: number, data: string, x: number, y: number, w: number, h: number): this;
  /** Place a QR-code image on the page (square: size × size pt). */
  barcodeQr(data: string, x: number, y: number, size: number): this;

  // --- Graphics primitives ------------------------------------------
  rect(x: number, y: number, w: number, h: number): this;
  filledRect(x: number, y: number, w: number, h: number, r: number, g: number, b: number): this;
  line(x1: number, y1: number, x2: number, y2: number): this;

  // --- v0.3.39 table primitives (#393) ------------------------------
  strokeRect(
    x: number,
    y: number,
    w: number,
    h: number,
    style?: { width?: number; color?: [number, number, number] }
  ): this;
  strokeLine(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    style?: { width?: number; color?: [number, number, number] }
  ): this;
  strokeRectDashed(
    x: number,
    y: number,
    w: number,
    h: number,
    dash: number[],
    phase?: number,
    style?: { width?: number; color?: [number, number, number] }
  ): this;
  strokeLineDashed(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    dash: number[],
    phase?: number,
    style?: { width?: number; color?: [number, number, number] }
  ): this;
  textInRect(x: number, y: number, w: number, h: number, text: string, align?: Align): this;
  newPageSameSize(): this;
  /** Approximate width of `text` in the current font (JS-side in v0.3.39). */
  measure(text: string): number;
  /** Remaining vertical space, or null when unknown. */
  remainingSpace(): number | null;
  /** Emit a buffered table at the current cursor. */
  table(spec: TableSpec): this;
  /** Begin a managed streaming-table adapter. */
  streamingTable(config: StreamingTableConfig): StreamingTable;

  // --- Lifecycle -----------------------------------------------------
  done(): DocumentBuilder;
  close(): void;
  [Symbol.dispose](): void;
}
