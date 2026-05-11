// SPDX-License-Identifier: MIT OR Apache-2.0
// PDF Oxide Node.js bindings - Native module loader

import { createRequire } from 'node:module';
import { dirname } from 'node:path';
import { arch, platform } from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  Align,
  AnnotationBuilder,
  ConversionOptionsBuilder,
  DocumentBuilder,
  EmbeddedFont,
  MetadataBuilder,
  PageBuilder,
  PdfBuilder,
  SearchOptionsBuilder,
  StreamingTable,
} from './builders/index';
import { DocumentEditor } from './document-editor';
import {
  AccessibilityException,
  CertificateLoadFailed,
  ComplianceException,
  EncryptionException,
  ErrorCategory,
  ErrorSeverity,
  InvalidStateException,
  IoException,
  mapFfiErrorCode,
  OcrException,
  OptimizationException,
  ParseException,
  PdfException,
  RedactionException,
  RenderingException,
  SearchException,
  SignatureException,
  SigningFailed,
  UnknownError,
  UnsupportedFeatureException,
  ValidationException,
  wrapAsyncMethod,
  wrapError,
  wrapMethod,
} from './errors';
import {
  AnnotationManager,
  type BatchDocument,
  BatchManager,
  type BatchOptions,
  type BatchProgress,
  type BatchResult,
  type BatchStatistics,
  createExtractionStream,
  createMetadataStream,
  createSearchStream,
  ExtractionManager,
  ExtractionStream,
  LayerManager,
  MetadataManager,
  MetadataStream,
  OutlineManager,
  RenderingManager,
  SearchManager,
  SearchStream,
  SecurityManager,
} from './managers/index';
import type {
  Column,
  SpanCell,
  StreamingTableConfig,
  Table,
  TableMode,
  TableSpec,
} from './types/common.js';
import type { WorkerResult, WorkerTask } from './workers/index';
import { WorkerPool, workerPool } from './workers/index';

// Create require function for CommonJS modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(import.meta.url);

// Phase 4+ managers (compiled JavaScript - use require for dynamic import)
// Phase 9: Now imports from canonical consolidated managers in managers/
const {
  OcrManager,
  OcrManager: OCRManager,
  OcrDetectionMode: OCRDetectionMode,
  ComplianceManager,
  PdfALevel,
  PdfXLevel,
  PdfUALevel,
  ComplianceIssueType,
  IssueSeverity,
  SignatureManager,
  SignatureAlgorithm,
  DigestAlgorithm,
  BarcodeManager,
  BarcodeFormat,
  BarcodeErrorCorrection,
  FormFieldManager,
  FormFieldType,
  FieldVisibility,
  ResultAccessorsManager,
  SearchResultProperties,
  FontProperties,
  ImageProperties,
  AnnotationProperties,
  ThumbnailManager,
  ThumbnailSize,
  ImageFormat,
  HybridMLManager,
  PageComplexity,
  ContentType,
  XfaManager,
  XfaFormType,
  XfaFieldType,
  CacheManager,
  EditingManager,
  AccessibilityManager,
  OptimizationManager,
  EnterpriseManager,
} = require('../lib/managers/index.js') as any;
// OcrLanguage re-exported from canonical OcrManager
const { OcrLanguage: OCRLanguage } = require('../lib/managers/ocr-manager.js') as any;

/**
 * Platform-specific prebuild paths (relative to compiled lib/index.js).
 * At runtime lib/index.js lives at js/lib/index.js, so ../prebuilds/
 * resolves to js/prebuilds/.
 */
const PLATFORMS: Record<string, Record<string, string>> = {
  darwin: {
    x64: '../prebuilds/darwin-x64/pdf_oxide.node',
    arm64: '../prebuilds/darwin-arm64/pdf_oxide.node',
  },
  linux: {
    x64: '../prebuilds/linux-x64/pdf_oxide.node',
    arm64: '../prebuilds/linux-arm64/pdf_oxide.node',
  },
  win32: {
    x64: '../prebuilds/win32-x64/pdf_oxide.node',
  },
};

/**
 * Gets the prebuild path for the current platform and architecture
 * @returns Path to the prebuild .node file (relative to lib/index.js)
 * @throws Error if platform or architecture is not supported
 */
function getPrebuildPath(): string {
  const osPaths = PLATFORMS[platform];
  if (!osPaths) {
    throw new Error(
      `Unsupported platform: ${platform}. Supported platforms: ${Object.keys(PLATFORMS).join(', ')}`
    );
  }

  const prebuildPath = osPaths[arch];
  if (!prebuildPath) {
    throw new Error(
      `Unsupported architecture: ${arch} for ${platform}. Supported architectures: ${Object.keys(osPaths).join(', ')}`
    );
  }

  return prebuildPath;
}

let nativeModule: any;

/**
 * Loads the native module dynamically based on platform and architecture.
 * Prebuilt .node files are bundled under prebuilds/<triple>/ in the package.
 * @returns Native module
 * @throws Error if native module cannot be loaded
 */
function loadNativeModule(): any {
  if (nativeModule) {
    return nativeModule;
  }

  try {
    const prebuildPath = getPrebuildPath();
    try {
      // Load the bundled prebuild .node file
      nativeModule = require(prebuildPath);
    } catch (e) {
      // Fallback to local build output if in development
      if (process.env.NODE_ENV === 'development' || process.env.NAPI_DEV) {
        try {
          nativeModule = require('./pdf-oxide');
        } catch {
          throw e;
        }
      } else {
        throw e;
      }
    }
    return nativeModule;
  } catch (error) {
    throw new Error(`Failed to load native module: ${(error as Error).message}`);
  }
}

// Load native module
const native = loadNativeModule();

/**
 * Wraps native class methods to convert errors to proper JavaScript Error subclasses.
 * This ensures that errors thrown from native code are instanceof the appropriate Error class.
 * @param nativeClass - The native class to wrap
 * @param asyncMethods - Names of async methods to wrap specially
 * @returns Wrapped class with error-handling methods
 */
function wrapNativeClass(nativeClass: any, asyncMethods: string[] = []): any {
  if (!nativeClass) return nativeClass;

  // For static methods like PdfDocument.open()
  for (const key of Object.getOwnPropertyNames(nativeClass)) {
    if (
      key !== 'prototype' &&
      key !== 'length' &&
      key !== 'name' &&
      typeof nativeClass[key] === 'function'
    ) {
      const isAsync = asyncMethods.includes(key);
      if (isAsync) {
        nativeClass[key] = wrapAsyncMethod(nativeClass[key], nativeClass);
      } else {
        nativeClass[key] = wrapMethod(nativeClass[key], nativeClass);
      }
    }
  }

  // For instance methods, wrap the prototype
  if (nativeClass.prototype) {
    for (const key of Object.getOwnPropertyNames(nativeClass.prototype)) {
      if (key !== 'constructor' && typeof nativeClass.prototype[key] === 'function') {
        const isAsync = asyncMethods.includes(key);
        const descriptor = Object.getOwnPropertyDescriptor(nativeClass.prototype, key);
        if (descriptor && descriptor.writable) {
          if (isAsync) {
            nativeClass.prototype[key] = wrapAsyncMethod(nativeClass.prototype[key]);
          } else {
            nativeClass.prototype[key] = wrapMethod(nativeClass.prototype[key]);
          }
        }
      }
    }
  }

  return nativeClass;
}

// ---------------------------------------------------------------------------
// JS wrapper classes around native loose-function exports.
//
// The binding.cc addon exports flat C functions (openDocument, extractText,
// pdfFromMarkdown, …) not N-API class constructors. These TS classes provide
// the idiomatic JS/TS API that users import. They mirror the Go binding's
// PdfDocument / PdfCreator / DocumentEditor pattern exactly — a handle-based
// lifecycle wrapping the same FFI surface.
// ---------------------------------------------------------------------------

/**
 * Options mirroring Rust's `RenderOptions` struct
 * (see `src/rendering/page_renderer.rs:41`). Used by
 * {@link PdfDocumentImpl.renderPageWithOptions}.
 */
export interface RenderOptions {
  /** Resolution (default 150). */
  dpi?: number;
  /** Output format (default PNG). */
  format?: 'png' | 'jpeg';
  /** RGBA 0..=1 tuple (default opaque white). */
  background?: [number, number, number, number];
  /** Drop background fill entirely (overrides `background`). */
  transparentBackground?: boolean;
  /** Render annotation layer (default true). */
  renderAnnotations?: boolean;
  /** JPEG quality 1..=100 (default 85). */
  jpegQuality?: number;
}

/**
 * Raw premultiplied RGBA8888 pixel buffer. Layout: row-major, top-left origin,
 * 4 bytes (R,G,B,A) per pixel, `data.length === width * height * 4`.
 * Alpha is premultiplied (PDF spec §11 transparency model).
 */
export interface RgbaPixmap {
  data: Buffer;
  width: number;
  height: number;
}

class PdfDocumentImpl {
  private _handle: any;
  private _closed = false;
  /** Promise-cached mutex; single Promise ensures no race on concurrent first calls. */
  private _muPromise: Promise<import('async-mutex').Mutex> | null = null;

  constructor(handle: any) {
    if (!handle) throw new Error('Failed to open document');
    this._handle = handle;
  }

  static open(path: string): PdfDocumentImpl {
    const handle = native.openDocument(path);
    return new PdfDocumentImpl(handle);
  }

  static openFromBuffer(buffer: Buffer | Uint8Array): PdfDocumentImpl {
    const handle = native.openFromBuffer(buffer);
    return new PdfDocumentImpl(handle);
  }

  static openWithPassword(path: string, password: string): PdfDocumentImpl {
    const handle = native.openWithPassword(path, password);
    return new PdfDocumentImpl(handle);
  }

  private ensureOpen(): void {
    if (this._closed) throw new Error('Document is closed');
  }

  get handle(): any {
    return this._handle;
  }

  pageCount(): number {
    this.ensureOpen();
    return native.getPageCount(this._handle);
  }
  getPageCount(): number {
    return this.pageCount();
  }
  get PageCount(): number {
    return this.pageCount();
  }

  extractText(pageIndex: number): string {
    this.ensureOpen();
    return native.extractText(this._handle, pageIndex);
  }
  toMarkdown(pageIndex: number): string {
    this.ensureOpen();
    return native.toMarkdown(this._handle, pageIndex);
  }
  toHtml(pageIndex: number): string {
    this.ensureOpen();
    return native.toHtml(this._handle, pageIndex);
  }
  toPlainText(pageIndex: number): string {
    this.ensureOpen();
    return native.toPlainText(this._handle, pageIndex);
  }
  toMarkdownAll(): string {
    this.ensureOpen();
    return native.toMarkdownAll(this._handle);
  }
  extractAllText(): string {
    this.ensureOpen();
    return native.extractAllText(this._handle);
  }
  toHtmlAll(): string {
    this.ensureOpen();
    return native.toHtmlAll(this._handle);
  }
  toPlainTextAll(): string {
    this.ensureOpen();
    return native.toPlainTextAll(this._handle);
  }

  getVersion(): { major: number; minor: number } {
    this.ensureOpen();
    return native.getVersion(this._handle);
  }
  hasStructureTree(): boolean {
    this.ensureOpen();
    return native.hasStructureTree(this._handle);
  }
  hasXFA(): boolean {
    this.ensureOpen();
    return native.hasXFA(this._handle);
  }

  getPageWidth(pageIndex: number): number {
    this.ensureOpen();
    return native.getPageWidth(this._handle, pageIndex);
  }
  getPageHeight(pageIndex: number): number {
    this.ensureOpen();
    return native.getPageHeight(this._handle, pageIndex);
  }
  getPageRotation(pageIndex: number): number {
    this.ensureOpen();
    return native.getPageRotation(this._handle, pageIndex);
  }

  searchPage(pageIndex: number, query: string, caseSensitive = false): any {
    this.ensureOpen();
    return native.searchPage(this._handle, pageIndex, query, caseSensitive);
  }

  searchAll(query: string, caseSensitive = false): any {
    this.ensureOpen();
    return native.searchAll(this._handle, query, caseSensitive);
  }

  getFormFields(): any {
    this.ensureOpen();
    return native.getFormFields(this._handle);
  }
  getOutline(): any {
    this.ensureOpen();
    return native.getOutline(this._handle);
  }
  getPageAnnotations(pageIndex: number): any {
    this.ensureOpen();
    return native.getPageAnnotations(this._handle, pageIndex);
  }
  getEmbeddedFonts(pageIndex: number): any {
    this.ensureOpen();
    return native.getEmbeddedFonts(this._handle, pageIndex);
  }
  getEmbeddedImages(pageIndex: number): any {
    this.ensureOpen();
    return native.getEmbeddedImages(this._handle, pageIndex);
  }
  extractWords(pageIndex: number): any {
    this.ensureOpen();
    return native.extractWords(this._handle, pageIndex);
  }
  extractTextLines(pageIndex: number): any {
    this.ensureOpen();
    return native.extractTextLines(this._handle, pageIndex);
  }
  extractTables(pageIndex: number): Table[] {
    this.ensureOpen();
    return native.extractTables(this._handle, pageIndex);
  }
  extractPaths(pageIndex: number): any {
    this.ensureOpen();
    return native.extractPaths(this._handle, pageIndex);
  }
  ocrExtractText(pageIndex: number, engineHandle: any): any {
    this.ensureOpen();
    return native.ocrExtractText(this._handle, pageIndex, engineHandle);
  }

  /**
   * Render a page with the full Rust `RenderOptions` surface
   * (DPI, format, RGBA background, transparency, annotation toggle,
   * JPEG quality). Returns the image bytes.
   */
  renderPageWithOptions(pageIndex: number, options: RenderOptions = {}): Uint8Array {
    this.ensureOpen();
    const dpi = options.dpi ?? 150;
    if (dpi <= 0) throw new RangeError(`dpi must be > 0, got ${dpi}`);
    const format = options.format === 'jpeg' ? 1 : 0;
    const quality = options.jpegQuality ?? 85;
    if (quality < 1 || quality > 100) {
      throw new RangeError(`jpegQuality must be in 1..=100, got ${quality}`);
    }
    const bg = options.background ?? [1, 1, 1, 1];
    const renderAnnotations = options.renderAnnotations === false ? 0 : 1;
    const transparent = options.transparentBackground ? 1 : 0;

    const imgHandle = native.renderPageWithOptions(
      this._handle,
      pageIndex,
      dpi,
      format,
      bg[0],
      bg[1],
      bg[2],
      bg[3],
      transparent,
      renderAnnotations,
      quality
    );
    try {
      const buf = native.pdfGetRenderedImageData(imgHandle);
      return new Uint8Array(buf);
    } finally {
      if (native.freeRenderedImage) {
        native.freeRenderedImage(imgHandle);
      }
    }
  }

  /**
   * Estimate render time (milliseconds) for a page at a given DPI.
   * Thin wrapper around the existing `estimateRenderTime` N-API
   * export — exposed in TS for the first time as part of gap L.
   */
  estimateRenderTime(pageIndex: number, dpi = 150): number {
    this.ensureOpen();
    return native.estimateRenderTime(this._handle, pageIndex, dpi);
  }

  /**
   * Render a page to fit inside a `width × height` pixel box, preserving
   * aspect ratio. Picks the largest DPI such that both rendered
   * dimensions are ≤ the target box, so the output may be smaller than
   * `width × height` on one axis. Issue #448.
   *
   * @param pageIndex   zero-based page index
   * @param width       target box width (pixels, must be > 0)
   * @param height      target box height (pixels, must be > 0)
   * @param format      `'png'` (default) or `'jpeg'`
   */
  renderPageFit(
    pageIndex: number,
    width: number,
    height: number,
    format: 'png' | 'jpeg' = 'png'
  ): Uint8Array {
    this.ensureOpen();
    if (width <= 0 || height <= 0) {
      throw new RangeError(`width and height must be > 0, got ${width}×${height}`);
    }
    const fmt = format === 'jpeg' ? 1 : 0;
    const imgHandle = native.renderPageFit(this._handle, pageIndex, width, height, fmt);
    try {
      const buf = native.pdfGetRenderedImageData(imgHandle);
      return new Uint8Array(buf);
    } finally {
      if (native.freeRenderedImage) {
        native.freeRenderedImage(imgHandle);
      }
    }
  }

  /** Lazy-initialises the per-instance Mutex (avoids a hard dep at load time). */
  private _getMutex(): Promise<import('async-mutex').Mutex> {
    if (!this._muPromise) {
      this._muPromise = import('async-mutex').then(({ Mutex }) => new Mutex());
    }
    return this._muPromise;
  }

  /**
   * Async variant of {@link renderPageWithOptions} serialised through a
   * per-instance mutex. Use when calling from `worker_threads` or `Promise.all`.
   */
  async renderPageWithOptionsAsync(
    pageIndex: number,
    options: RenderOptions = {}
  ): Promise<Uint8Array> {
    const mu = await this._getMutex();
    return mu.runExclusive(() => this.renderPageWithOptions(pageIndex, options));
  }

  /**
   * Async variant of {@link renderPageFit} serialised through a per-instance mutex.
   */
  async renderPageFitAsync(
    pageIndex: number,
    width: number,
    height: number,
    format: 'png' | 'jpeg' = 'png'
  ): Promise<Uint8Array> {
    const mu = await this._getMutex();
    return mu.runExclusive(() => this.renderPageFit(pageIndex, width, height, format));
  }

  /**
   * Renders a page as raw premultiplied RGBA8888 pixels. No PNG/JPEG encoding
   * overhead — useful for direct handoff to image-processing libraries.
   * @param pageIndex Zero-based page index.
   * @param dpi Resolution in dots per inch (default 150).
   */
  renderToPixmap(pageIndex: number, dpi = 150): RgbaPixmap {
    this.ensureOpen();
    if (dpi <= 0) throw new RangeError(`dpi must be > 0, got ${dpi}`);
    const { imgHandle, width, height } = native.renderPageRaw(this._handle, pageIndex, dpi);
    try {
      const buf = native.pdfGetRenderedImageData(imgHandle);
      return { data: Buffer.from(buf), width, height };
    } finally {
      if (native.freeRenderedImage) native.freeRenderedImage(imgHandle);
    }
  }

  /**
   * Async variant of {@link renderToPixmap} serialised through a per-instance mutex.
   */
  async renderToPixmapAsync(pageIndex: number, dpi = 150): Promise<RgbaPixmap> {
    const mu = await this._getMutex();
    return mu.runExclusive(() => this.renderToPixmap(pageIndex, dpi));
  }

  page(index: number): Page {
    this.ensureOpen();
    const count = this.pageCount();
    const idx = index < 0 ? count + index : index;
    if (idx < 0 || idx >= count) throw new RangeError(`page index ${index} out of range`);
    return new Page(this, idx);
  }

  [Symbol.iterator](): Iterator<Page> {
    this.ensureOpen();
    const count = this.pageCount();
    let i = 0;
    const doc = this;
    return {
      next(): IteratorResult<Page> {
        if (i >= count) return { value: undefined as any, done: true };
        return { value: new Page(doc, i++), done: false };
      },
    };
  }

  /**
   * Async iteration over the document's pages — issue #447. The body
   * is identical to the sync iterator (page handles are constructed
   * synchronously) but exposing this surface lets consumers `for await`
   * uniformly with other async resources without an explicit
   * `Promise.resolve(...)`.
   */
  [Symbol.asyncIterator](): AsyncIterator<Page> {
    this.ensureOpen();
    const count = this.pageCount();
    let i = 0;
    const doc = this;
    return {
      async next(): Promise<IteratorResult<Page>> {
        if (i >= count) return { value: undefined as any, done: true };
        return { value: new Page(doc, i++), done: false };
      },
    };
  }

  /**
   * Validate PDF/A conformance at a given level.
   * @param level - "1a"|"1b"|"2a"|"2b"|"2u"|"3a"|"3b"|"3u" (default "2b")
   */
  validatePdfA(level: '1a' | '1b' | '2a' | '2b' | '2u' | '3a' | '3b' | '3u' = '2b'): {
    compliant: boolean;
    errors: string[];
    warnings: string[];
  } {
    this.ensureOpen();
    const levelMap: Record<string, number> = {
      '1b': 0,
      '1a': 1,
      '2b': 2,
      '2a': 3,
      '2u': 4,
      '3b': 5,
      '3a': 6,
      '3u': 7,
    };
    const levelInt = levelMap[level];
    if (levelInt === undefined) throw new RangeError(`Unknown PDF/A level: "${level}"`);
    return native.validatePdfALevel(this._handle, levelInt);
  }

  /**
   * Convert document to PDF/A conformance in-place.
   * @param level - "1b"|"2b"|"2u"|"3b" etc. (default "2b")
   * @returns true if the document is fully PDF/A-compliant after conversion (false if errors remain, e.g. fonts not embeddable without the rendering feature)
   */
  convertToPdfA(level: '1a' | '1b' | '2a' | '2b' | '2u' | '3a' | '3b' | '3u' = '2b'): boolean {
    this.ensureOpen();
    const levelMap: Record<string, number> = {
      '1b': 0,
      '1a': 1,
      '2b': 2,
      '2a': 3,
      '2u': 4,
      '3b': 5,
      '3a': 6,
      '3u': 7,
    };
    const levelInt = levelMap[level];
    if (levelInt === undefined) throw new RangeError(`Unknown PDF/A level: "${level}"`);
    return native.convertToPdfA(this._handle, levelInt);
  }

  /**
   * Return the current document bytes (including any in-place modifications
   * made by convertToPdfA).
   */
  toBuffer(): Buffer {
    this.ensureOpen();
    return native.documentGetSourceBytes(this._handle);
  }

  close(): void {
    if (!this._closed && this._handle) {
      native.closeDocument(this._handle);
      this._closed = true;
    }
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

class Page {
  private _doc: PdfDocumentImpl;
  private _index: number;
  private _cache: Map<string, any> = new Map();

  constructor(doc: PdfDocumentImpl, index: number) {
    this._doc = doc;
    this._index = index;
  }

  get index(): number {
    return this._index;
  }

  get width(): number {
    if (!this._cache.has('width')) this._cache.set('width', this._doc.getPageWidth(this._index));
    return this._cache.get('width');
  }

  get height(): number {
    if (!this._cache.has('height')) this._cache.set('height', this._doc.getPageHeight(this._index));
    return this._cache.get('height');
  }

  get rotation(): number {
    if (!this._cache.has('rotation'))
      this._cache.set('rotation', this._doc.getPageRotation(this._index));
    return this._cache.get('rotation');
  }

  text(): string {
    return this._doc.extractText(this._index);
  }
  markdown(): string {
    return this._doc.toMarkdown(this._index);
  }
  html(): string {
    return this._doc.toHtml(this._index);
  }
  plainText(): string {
    return this._doc.toPlainText(this._index);
  }
  words(): any {
    return this._doc.extractWords(this._index);
  }
  lines(): any {
    return this._doc.extractTextLines(this._index);
  }
  tables(): Table[] {
    return this._doc.extractTables(this._index);
  }
  images(): any {
    return this._doc.getEmbeddedImages(this._index);
  }
  paths(): any {
    return this._doc.extractPaths(this._index);
  }
  annotations(): any {
    return this._doc.getPageAnnotations(this._index);
  }
  fonts(): any {
    return this._doc.getEmbeddedFonts(this._index);
  }
  search(query: string, caseSensitive = false): any {
    return this._doc.searchPage(this._index, query, caseSensitive);
  }

  toString(): string {
    return `Page(index=${this._index})`;
  }
}

class PdfImpl {
  private _handle: any;
  private _closed = false;

  constructor(handle: any) {
    if (!handle) throw new Error('Failed to create PDF');
    this._handle = handle;
  }

  static fromMarkdown(markdown: string): PdfImpl {
    return new PdfImpl(native.pdfFromMarkdown(markdown));
  }

  static fromHtml(html: string): PdfImpl {
    return new PdfImpl(native.pdfFromHtml(html));
  }

  static fromText(text: string): PdfImpl {
    return new PdfImpl(native.pdfFromText(text));
  }

  static fromImage(path: string): PdfImpl {
    return new PdfImpl(native.pdfFromImage(path));
  }

  static fromImageBytes(data: Buffer | Uint8Array): PdfImpl {
    return new PdfImpl(native.pdfFromImageBytes(data));
  }

  static fromHtmlCss(html: string, css: string, fontBytes: Buffer | Uint8Array): PdfImpl {
    return new PdfImpl(native.pdfFromHtmlCss(html, css, fontBytes));
  }

  static fromHtmlCssWithFonts(
    html: string,
    css: string,
    families: string[],
    fonts: (Buffer | Uint8Array)[]
  ): PdfImpl {
    if (families.length !== fonts.length) {
      throw new Error(
        `fromHtmlCssWithFonts: families.length (${families.length}) must equal fonts.length (${fonts.length})`
      );
    }
    return new PdfImpl(native.pdfFromHtmlCssWithFonts(html, css, families, fonts));
  }

  private ensureOpen(): void {
    if (this._closed) throw new Error('PDF handle is closed');
  }

  save(path: string): void {
    this.ensureOpen();
    native.pdfSave(this._handle, path);
  }
  saveToBytes(): Buffer {
    this.ensureOpen();
    return native.pdfSaveToBytes(this._handle);
  }
  pageCount(): number {
    this.ensureOpen();
    return native.pdfGetPageCount(this._handle);
  }

  close(): void {
    if (!this._closed && this._handle) {
      native.pdfFree(this._handle);
      this._closed = true;
    }
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

// Generate a 1D barcode as a vector SVG string.
// format: 0=Code128, 1=Code39, 2=EAN13, 3=EAN8, 4=UPCA, 5=ITF, 6=Code93, 7=Codabar.
function generateBarcodeSvg(data: string, format: number = 0, sizePx: number = 300): string {
  const handle = native.generateBarcode(format, data);
  try {
    return native.barcodeGetSVG(handle, sizePx) as string;
  } finally {
    native.freeBarcode(handle);
  }
}

// Generate a QR code as a vector SVG string.
// errorCorrection: 0=Low, 1=Medium, 2=Quartile, 3=High.
function generateQrCodeSvg(
  data: string,
  errorCorrection: number = 1,
  sizePx: number = 300
): string {
  const handle = native.generateQRCode(data, errorCorrection);
  try {
    return native.barcodeGetSVG(handle, sizePx) as string;
  } finally {
    native.freeBarcode(handle);
  }
}

// Export as ES module
const getVersion = native.getVersion;
const getPdfOxideVersion = native.getPdfOxideVersion;
const getActiveCryptoProvider = native.getActiveCryptoProvider;
const isFipsCryptoAvailable = native.isFipsCryptoAvailable;
const useFipsCryptoProvider = native.useFipsCryptoProvider;
const PdfDocument = PdfDocumentImpl as any;
const Pdf = PdfImpl as any;
const PdfError = PdfException;
const PageSize = native.PageSize;
const Rect = native.Rect;
const Point = native.Point;
const Color = native.Color;
const ConversionOptions = native.ConversionOptions;
const SearchOptions = native.SearchOptions;
const SearchResult = native.SearchResult;
const TextSearcher = native.TextSearcher;

// RFC 3161 Timestamp + TSA Client — standalone, re-exported from
// their own modules so downstream users get the full API surface.
export { Timestamp, TimestampHashAlgorithm } from './timestamp.js';
export { TsaClient, type TsaClientOptions } from './tsa-client.js';
export type {
  BatchDocument,
  BatchOptions,
  BatchProgress,
  BatchResult,
  BatchStatistics,
  Column,
  SpanCell,
  StreamingTableConfig,
  Table,
  TableMode,
  TableSpec,
  WorkerResult,
  WorkerTask,
};
export {
  AccessibilityException,
  AccessibilityManager,
  // v0.3.39 — DocumentBuilder tables (#393)
  Align,
  AnnotationBuilder,
  AnnotationManager,
  AnnotationProperties,
  BarcodeErrorCorrection,
  BarcodeFormat,
  BarcodeManager,
  // Phase 2.5: Batch Processing API
  BatchManager,
  CacheManager,
  CertificateLoadFailed,
  Color,
  ComplianceException,
  ComplianceIssueType,
  ComplianceManager,
  ContentType,
  ConversionOptions,
  ConversionOptionsBuilder,
  createExtractionStream,
  createMetadataStream,
  createSearchStream,
  DigestAlgorithm,
  // Write-side fluent API
  DocumentBuilder,
  // Editor mutation API
  DocumentEditor,
  EditingManager,
  EmbeddedFont,
  EncryptionException,
  EnterpriseManager,
  // Error utilities
  ErrorCategory,
  ErrorSeverity,
  ExtractionManager,
  ExtractionStream,
  FieldVisibility,
  FontProperties,
  FormFieldManager,
  FormFieldType,
  generateBarcodeSvg,
  generateQrCodeSvg,
  getActiveCryptoProvider,
  getPdfOxideVersion,
  // Version info
  getVersion,
  HybridMLManager,
  ImageFormat,
  ImageProperties,
  InvalidStateException,
  IoException,
  IssueSeverity,
  isFipsCryptoAvailable,
  LayerManager,
  MetadataBuilder,
  MetadataManager,
  MetadataStream,
  mapFfiErrorCode,
  OCRDetectionMode,
  OCRLanguage,
  OCRManager,
  OcrException,
  // Managers (Phase 4+, consolidated in Phase 9)
  OcrManager,
  OptimizationException,
  OptimizationManager,
  // Managers (Phase 1-3: Core)
  OutlineManager,
  Page,
  PageBuilder,
  PageComplexity,
  // Types
  PageSize,
  ParseException,
  Pdf,
  PdfALevel,
  // Builders
  PdfBuilder,
  // Main classes
  PdfDocument,
  // Error types
  PdfError,
  PdfException,
  PdfUALevel,
  PdfXLevel,
  Point,
  Rect,
  RedactionException,
  RenderingException,
  RenderingManager,
  ResultAccessorsManager,
  SearchException,
  SearchManager,
  SearchOptions,
  SearchOptionsBuilder,
  SearchResult,
  SearchResultProperties,
  // Phase 2.4: Stream API
  SearchStream,
  SecurityManager,
  SignatureAlgorithm,
  SignatureException,
  SignatureManager,
  SigningFailed,
  // v0.3.39 — managed streaming-table adapter (#393)
  StreamingTable,
  // Utilities
  TextSearcher,
  ThumbnailManager,
  ThumbnailSize,
  UnknownError,
  UnsupportedFeatureException,
  useFipsCryptoProvider,
  ValidationException,
  // Worker Threads API
  WorkerPool,
  workerPool,
  wrapAsyncMethod,
  wrapError,
  wrapMethod,
  XfaFieldType,
  XfaFormType,
  XfaManager,
};
