// pdf_oxide — idiomatic Objective-C bindings over the C ABI.
//
// NSObject wrappers (POXDocument, POXPdf) own the C handles and free them in
// -dealloc; returned C strings/buffers are copied into NSString/NSData and
// freed via free_string; non-success C-ABI error codes surface as NSError
// (POXErrorDomain). API surface mirrors the other language bindings; coverage
// is asserted by POXApiCoverageTests (one test per method).
#import <Foundation/Foundation.h>

/// Binding version, kept in lock-step with the workspace crate by
/// scripts/sync_version.py (the single source of truth is Cargo.toml).
#define POX_PDF_OXIDE_VERSION "0.3.69"

NS_ASSUME_NONNULL_BEGIN

extern NSString* const POXErrorDomain;

@class POXCertificate;
@class POXTimestamp;
@class POXPdfAResults;
@class POXUaResults;
@class POXPdfXResults;
@class POXBarcode;
@class POXOcrEngine;
@class POXElementList;
@class POXSignatureInfo;
@class POXDss;
@class POXDocument;

/// PDF version with named major/minor fields.
typedef struct {
    uint8_t major;
    uint8_t minor;
} POXVersion;

/// A bounding box in PDF user-space units (origin/size).
typedef struct {
    float x;
    float y;
    float width;
    float height;
} POXBbox;

@class POXPage;

/// A single extracted character (Phase-1 element extraction).
@interface POXChar : NSObject
/// The Unicode codepoint of the character.
@property(nonatomic, readonly) uint32_t character;
@property(nonatomic, readonly) POXBbox bbox;
@property(nonatomic, readonly, copy) NSString* fontName;
@property(nonatomic, readonly) float fontSize;
@end

/// A single extracted word (Phase-1 element extraction).
@interface POXWord : NSObject
@property(nonatomic, readonly, copy) NSString* text;
@property(nonatomic, readonly) POXBbox bbox;
@property(nonatomic, readonly, copy) NSString* fontName;
@property(nonatomic, readonly) float fontSize;
@property(nonatomic, readonly) BOOL bold;
@end

/// A single extracted text line (Phase-1 element extraction).
@interface POXTextLine : NSObject
@property(nonatomic, readonly, copy) NSString* text;
@property(nonatomic, readonly) POXBbox bbox;
@property(nonatomic, readonly) NSInteger wordCount;
@end

/// A single extracted table (Phase-1 element extraction).
@interface POXTable : NSObject
@property(nonatomic, readonly) NSInteger rowCount;
@property(nonatomic, readonly) NSInteger colCount;
@property(nonatomic, readonly) BOOL hasHeader;
/// Cell text at (row, col); nil if out of range or unavailable.
- (nullable NSString*)cellTextAtRow:(NSInteger)row col:(NSInteger)col;
@end

/// A single embedded font (Phase-2 extraction).
@interface POXFont : NSObject
@property(nonatomic, readonly, copy) NSString* name;
@property(nonatomic, readonly, copy) NSString* type;
@property(nonatomic, readonly, copy) NSString* encoding;
@property(nonatomic, readonly) BOOL embedded;
@property(nonatomic, readonly) BOOL subset;
/// Nominal font size in points (0 if unknown).
@property(nonatomic, readonly) float size;
@end

/// A single embedded image (Phase-2 extraction).
@interface POXImage : NSObject
@property(nonatomic, readonly) NSInteger width;
@property(nonatomic, readonly) NSInteger height;
@property(nonatomic, readonly) NSInteger bitsPerComponent;
@property(nonatomic, readonly, copy) NSString* format;
@property(nonatomic, readonly, copy) NSString* colorspace;
@property(nonatomic, readonly, copy) NSData* data;
@end

/// A single page annotation (Phase-2 extraction).
@interface POXAnnotation : NSObject
@property(nonatomic, readonly, copy) NSString* type;
@property(nonatomic, readonly, copy) NSString* subtype;
@property(nonatomic, readonly, copy) NSString* content;
@property(nonatomic, readonly, copy) NSString* author;
@property(nonatomic, readonly) POXBbox rect;
@property(nonatomic, readonly) float borderWidth;
/// Annotation color packed as 0xAARRGGBB (0 if absent).
@property(nonatomic, readonly) uint32_t color;
/// Creation / modification time as Unix epoch seconds (0 if absent).
@property(nonatomic, readonly) int64_t creationDate;
@property(nonatomic, readonly) int64_t modificationDate;
@property(nonatomic, readonly) BOOL hidden;
@property(nonatomic, readonly) BOOL markedDeleted;
@property(nonatomic, readonly) BOOL printable;
@property(nonatomic, readonly) BOOL readOnly;
/// For Link annotations: the target URI (nil if none).
@property(nonatomic, readonly, copy, nullable) NSString* linkUri;
/// For Text (note) annotations: the icon name (nil if none).
@property(nonatomic, readonly, copy, nullable) NSString* iconName;
/// For Highlight annotations: an array of quad points, each an 8-element
/// NSArray<NSNumber*> {x1,y1,x2,y2,x3,y3,x4,y4}. Empty when not a highlight.
@property(nonatomic, readonly, copy) NSArray<NSArray<NSNumber*>*>* quadPoints;
@end

/// An AcroForm field (name / value / type, plus the readonly & required flags).
@interface POXFormField : NSObject
@property(nonatomic, readonly, copy) NSString* name;
@property(nonatomic, readonly, copy) NSString* value;
@property(nonatomic, readonly, copy) NSString* type;
@property(nonatomic, readonly) BOOL readonly;
@property(nonatomic, readonly) BOOL required;
@end

/// A single vector path (Phase-2 extraction).
@interface POXPath : NSObject
@property(nonatomic, readonly) POXBbox bbox;
@property(nonatomic, readonly) float strokeWidth;
@property(nonatomic, readonly) BOOL hasStroke;
@property(nonatomic, readonly) BOOL hasFill;
@property(nonatomic, readonly) NSInteger operationCount;
@end

/// A single text search result (Phase-2 extraction).
@interface POXSearchResult : NSObject
@property(nonatomic, readonly, copy) NSString* text;
@property(nonatomic, readonly) NSInteger page;
@property(nonatomic, readonly) POXBbox bbox;
@end

/// A rendered page image (Phase-3 rendering). Owns the native handle and frees
/// it on -close/-dealloc; width/height/data are read eagerly, and -saveToPath:
/// uses the live native handle.
@interface POXRenderedImage : NSObject
@property(nonatomic, readonly) NSInteger width;
@property(nonatomic, readonly) NSInteger height;
/// Encoded image bytes (e.g. PNG).
@property(nonatomic, readonly, copy) NSData* data;
/// Write the rendered image to a file; returns YES on success.
- (BOOL)saveToPath:(NSString*)path error:(NSError**)error;
/// Free the native handle now (idempotent).
- (void)close;
@end

/// An opened PDF for extraction/inspection.
@interface POXDocument : NSObject

/// Open a PDF from a filesystem path.
+ (nullable instancetype)openPath:(NSString*)path error:(NSError**)error;
/// Open a PDF from in-memory bytes.
+ (nullable instancetype)openFromBytes:(NSData*)data error:(NSError**)error;
/// Open a password-protected PDF.
+ (nullable instancetype)openWithPassword:(NSString*)path
                                 password:(NSString*)password
                                    error:(NSError**)error;

/// Convert Office documents (in-memory bytes) to a PDF document.
+ (nullable instancetype)openFromDocxBytes:(NSData*)data error:(NSError**)error;
+ (nullable instancetype)openFromPptxBytes:(NSData*)data error:(NSError**)error;
+ (nullable instancetype)openFromXlsxBytes:(NSData*)data error:(NSError**)error;

/// Export the document to an Office format (owned bytes).
- (nullable NSData*)toDocxWithError:(NSError**)error;
- (nullable NSData*)toPptxWithError:(NSError**)error;
- (nullable NSData*)toXlsxWithError:(NSError**)error;

/// Number of pages, or -1 on error (sets `error`).
- (NSInteger)pageCountError:(NSError**)error;
/// PDF version as a POXVersion {major, minor}.
- (POXVersion)version;
- (BOOL)isEncrypted;
- (BOOL)hasStructureTree;

- (nullable NSString*)extractText:(NSInteger)page error:(NSError**)error;
- (nullable NSString*)toPlainText:(NSInteger)page error:(NSError**)error;
- (nullable NSString*)toMarkdown:(NSInteger)page error:(NSError**)error;
- (nullable NSString*)toHtml:(NSInteger)page error:(NSError**)error;
- (nullable NSString*)toMarkdownAllWithError:(NSError**)error;
- (nullable NSString*)toHtmlAllWithError:(NSError**)error;
- (nullable NSString*)toPlainTextAllWithError:(NSError**)error;
- (nullable NSString*)extractStructuredJson:(NSInteger)page error:(NSError**)error;

/// Phase-1 element extraction (page index is 0-based).
- (nullable NSArray<POXChar*>*)extractChars:(NSInteger)page error:(NSError**)error;
- (nullable NSArray<POXWord*>*)extractWords:(NSInteger)page error:(NSError**)error;
- (nullable NSArray<POXTextLine*>*)extractTextLines:(NSInteger)page
                                              error:(NSError**)error;
- (nullable NSArray<POXTable*>*)extractTables:(NSInteger)page error:(NSError**)error;

/// Phase-2 extraction (page index is 0-based).
- (nullable NSArray<POXFont*>*)embeddedFonts:(NSInteger)page error:(NSError**)error;
- (nullable NSArray<POXImage*>*)embeddedImages:(NSInteger)page error:(NSError**)error;
- (nullable NSArray<POXAnnotation*>*)pageAnnotations:(NSInteger)page
                                               error:(NSError**)error;
- (nullable NSArray<POXPath*>*)extractPaths:(NSInteger)page error:(NSError**)error;
- (nullable NSArray<POXSearchResult*>*)search:(NSInteger)page
                                         term:(NSString*)term
                                caseSensitive:(BOOL)caseSensitive
                                        error:(NSError**)error;
- (nullable NSArray<POXSearchResult*>*)searchAll:(NSString*)term
                                   caseSensitive:(BOOL)caseSensitive
                                           error:(NSError**)error;

/// Phase-3 page rendering (page index is 0-based; format 0 = PNG).
- (nullable POXRenderedImage*)renderPage:(NSInteger)pageIndex
                                  format:(int32_t)format
                                   error:(NSError**)error;
- (nullable POXRenderedImage*)renderPageZoom:(NSInteger)pageIndex
                                        zoom:(float)zoom
                                      format:(int32_t)format
                                       error:(NSError**)error;
- (nullable POXRenderedImage*)renderPageThumbnail:(NSInteger)pageIndex
                                             size:(int32_t)size
                                           format:(int32_t)format
                                            error:(NSError**)error;

/// Authenticate a password-protected PDF; returns YES on success, NO for a
/// wrong password (no error). Sets `error` only on a genuine failure.
- (BOOL)authenticate:(NSString*)password error:(NSError**)error;

/// A page handle bound to this document (0-based). The page keeps the document
/// alive for as long as it lives.
- (POXPage*)pageAtIndex:(NSInteger)index;

/// Phase-6 conformance validation. Each returns a result handle (nil on a
/// genuine failure, which sets `error`). `level` selects the conformance
/// sub-level (e.g. PDF/A 0=A1b…7=A3u; PDF/UA 1; PDF/X by part).
- (nullable POXPdfAResults*)validatePdfA:(int32_t)level error:(NSError**)error;
- (nullable POXUaResults*)validatePdfUa:(int32_t)level error:(NSError**)error;
- (nullable POXPdfXResults*)validatePdfX:(int32_t)level error:(NSError**)error;

// ── Phase-7: render variants / page getters / OCR ────────────────────────────

/// Render a page with the full RenderOptions surface (background RGBA channels
/// 0.0–1.0; `transparentBackground`/`renderAnnotations` are 0/1 flags; format
/// 0=PNG 1=JPEG). Returns nil on a genuine failure (sets `error`).
- (nullable POXRenderedImage*)renderPageWithOptions:(NSInteger)pageIndex
                                                dpi:(int32_t)dpi
                                             format:(int32_t)format
                                                bgR:(float)bgR
                                                bgG:(float)bgG
                                                bgB:(float)bgB
                                                bgA:(float)bgA
                              transparentBackground:(int32_t)transparentBackground
                                  renderAnnotations:(int32_t)renderAnnotations
                                        jpegQuality:(int32_t)jpegQuality
                                              error:(NSError**)error;
/// Like -renderPageWithOptions: but also suppresses the named OCG layers.
- (nullable POXRenderedImage*)renderPageWithOptionsEx:(NSInteger)pageIndex
                                                  dpi:(int32_t)dpi
                                               format:(int32_t)format
                                                  bgR:(float)bgR
                                                  bgG:(float)bgG
                                                  bgB:(float)bgB
                                                  bgA:(float)bgA
                                transparentBackground:(int32_t)transparentBackground
                                    renderAnnotations:(int32_t)renderAnnotations
                                          jpegQuality:(int32_t)jpegQuality
                                       excludedLayers:
                                           (NSArray<NSString*>*)excludedLayers
                                                error:(NSError**)error;
/// Render a rectangular region (user-space points, origin bottom-left).
- (nullable POXRenderedImage*)renderPageRegion:(NSInteger)pageIndex
                                         cropX:(float)cropX
                                         cropY:(float)cropY
                                     cropWidth:(float)cropWidth
                                    cropHeight:(float)cropHeight
                                        format:(int32_t)format
                                         error:(NSError**)error;
/// Render a page to fit inside w×h pixels, preserving aspect ratio.
- (nullable POXRenderedImage*)renderPageFit:(NSInteger)pageIndex
                                          w:(int32_t)w
                                          h:(int32_t)h
                                     format:(int32_t)format
                                      error:(NSError**)error;
/// Render a page to a raw RGBA8888 buffer; on success `outWidth`/`outHeight`
/// receive the pixel dimensions.
- (nullable POXRenderedImage*)renderPageRaw:(NSInteger)pageIndex
                                        dpi:(int32_t)dpi
                                   outWidth:(int32_t*)outWidth
                                  outHeight:(int32_t*)outHeight
                                      error:(NSError**)error;
/// Estimate the render time (engine units) for a page, or -1 on error.
- (int32_t)estimateRenderTime:(NSInteger)pageIndex error:(NSError**)error;

/// Page geometry getters (0-based page index).
- (float)pageWidth:(NSInteger)pageIndex error:(NSError**)error;
- (float)pageHeight:(NSInteger)pageIndex error:(NSError**)error;
/// Page rotation in degrees, or -1 on error.
- (int32_t)pageRotation:(NSInteger)pageIndex error:(NSError**)error;
/// Layout-element list for a page (opaque handle owning the native list).
- (nullable POXElementList*)pageElements:(NSInteger)pageIndex error:(NSError**)error;

/// OCR-augmented text extraction. `engine` may be nil for native-only text.
- (BOOL)pageNeedsOcr:(NSInteger)pageIndex error:(NSError**)error;
- (nullable NSString*)ocrExtractText:(NSInteger)pageIndex
                              engine:(nullable POXOcrEngine*)engine
                               error:(NSError**)error;

// ── In-rect extractors (page-index 0-based; rect in user-space points) ────────
- (nullable NSString*)extractTextInRect:(NSInteger)page
                                      x:(float)x
                                      y:(float)y
                                  width:(float)width
                                 height:(float)height
                                  error:(NSError**)error;
- (nullable NSArray<POXWord*>*)extractWordsInRect:(NSInteger)page
                                                x:(float)x
                                                y:(float)y
                                            width:(float)width
                                           height:(float)height
                                            error:(NSError**)error;
- (nullable NSArray<POXTextLine*>*)extractLinesInRect:(NSInteger)page
                                                    x:(float)x
                                                    y:(float)y
                                                width:(float)width
                                               height:(float)height
                                                error:(NSError**)error;
- (nullable NSArray<POXTable*>*)extractTablesInRect:(NSInteger)page
                                                  x:(float)x
                                                  y:(float)y
                                              width:(float)width
                                             height:(float)height
                                              error:(NSError**)error;
- (nullable NSArray<POXImage*>*)extractImagesInRect:(NSInteger)page
                                                  x:(float)x
                                                  y:(float)y
                                              width:(float)width
                                             height:(float)height
                                              error:(NSError**)error;

// ── Auto extraction / classification ─────────────────────────────────────────
- (nullable NSString*)extractTextAuto:(NSInteger)page error:(NSError**)error;
- (nullable NSString*)extractAllTextWithError:(NSError**)error;
- (nullable NSString*)extractPageAuto:(NSInteger)page
                          optionsJson:(nullable NSString*)optionsJson
                                error:(NSError**)error;
- (nullable NSString*)classifyPage:(NSInteger)page error:(NSError**)error;
- (nullable NSString*)classifyDocumentWithError:(NSError**)error;

// ── Header / footer / artifact removal ───────────────────────────────────────
- (int32_t)eraseHeader:(NSInteger)page error:(NSError**)error;
- (int32_t)eraseFooter:(NSInteger)page error:(NSError**)error;
- (int32_t)eraseArtifacts:(NSInteger)page error:(NSError**)error;
- (int32_t)removeHeaders:(float)threshold error:(NSError**)error;
- (int32_t)removeFooters:(float)threshold error:(NSError**)error;
- (int32_t)removeArtifacts:(float)threshold error:(NSError**)error;

// ── AcroForm fields & form data ──────────────────────────────────────────────
- (nullable NSArray<POXFormField*>*)formFieldsWithError:(NSError**)error;
- (nullable NSData*)exportFormDataToBytes:(int32_t)formatType error:(NSError**)error;
- (BOOL)importFormDataFromPath:(NSString*)dataPath error:(NSError**)error;
- (BOOL)importFormFromFile:(NSString*)filename error:(NSError**)error;

// ── Document structure / metadata ────────────────────────────────────────────
- (nullable NSString*)outlineWithError:(NSError**)error;
- (nullable NSString*)pageLabelsWithError:(NSError**)error;
- (nullable NSString*)xmpMetadataWithError:(NSError**)error;
- (nullable NSData*)sourceBytesWithError:(NSError**)error;
- (BOOL)hasXfa;
- (nullable NSString*)planSplitByBookmarks:(nullable NSString*)optionsJson
                                     error:(NSError**)error;

// ── PDF/A conversion ─────────────────────────────────────────────────────────
- (BOOL)convertToPdfA:(int32_t)level error:(NSError**)error;

// ── Signatures (document-level) ──────────────────────────────────────────────
- (BOOL)sign:(POXCertificate*)certificate
      reason:(nullable NSString*)reason
    location:(nullable NSString*)location
       error:(NSError**)error;
- (int32_t)signatureCountWithError:(NSError**)error;
- (nullable POXSignatureInfo*)signatureAtIndex:(int32_t)index error:(NSError**)error;
- (int32_t)verifyAllSignaturesWithError:(NSError**)error;
- (int32_t)hasTimestampWithError:(NSError**)error;
- (nullable POXDss*)dssWithError:(NSError**)error;

// ── Annotations as JSON ──────────────────────────────────────────────────────
- (nullable NSString*)annotationsJson:(NSInteger)page error:(NSError**)error;

// ── Fonts / search results as JSON ───────────────────────────────────────────
- (nullable NSString*)embeddedFontsJson:(NSInteger)page error:(NSError**)error;
- (nullable NSString*)searchJson:(NSInteger)page
                            term:(NSString*)term
                   caseSensitive:(BOOL)caseSensitive
                           error:(NSError**)error;

/// Free the native handle now (idempotent).
- (void)close;

@end

/// A page bound to its POXDocument (0-based). Holds a strong reference to the
/// document so it cannot outlive it; each method delegates to the corresponding
/// per-page POXDocument method with the stored index.
@interface POXPage : NSObject

- (nullable NSString*)text:(NSError**)error;
- (nullable NSString*)markdown:(NSError**)error;
- (nullable NSString*)html:(NSError**)error;
- (nullable NSString*)plainText:(NSError**)error;

@end

/// A PDF produced by a builder.
@interface POXPdf : NSObject

+ (nullable instancetype)fromMarkdown:(NSString*)markdown error:(NSError**)error;
+ (nullable instancetype)fromHtml:(NSString*)html error:(NSError**)error;
+ (nullable instancetype)fromText:(NSString*)text error:(NSError**)error;

// ── Phase-7: image / HTML+CSS constructors ───────────────────────────────────

/// Build a single-page PDF wrapping an image file.
+ (nullable instancetype)fromImage:(NSString*)path error:(NSError**)error;
/// Build a single-page PDF wrapping in-memory image bytes.
+ (nullable instancetype)fromImageBytes:(NSData*)data error:(NSError**)error;
/// Build a PDF from HTML + CSS with a single optional embedded font (pass nil
/// `fontBytes` for none).
+ (nullable instancetype)fromHtml:(NSString*)html
                              css:(NSString*)css
                        fontBytes:(nullable NSData*)fontBytes
                            error:(NSError**)error;
/// Build a PDF from HTML + CSS with a multi-font cascade. `families` and
/// `fonts` are parallel arrays (families[i] names the family for fonts[i]).
+ (nullable instancetype)fromHtml:(NSString*)html
                              css:(NSString*)css
                         families:(NSArray<NSString*>*)families
                            fonts:(NSArray<NSData*>*)fonts
                            error:(NSError**)error;

- (BOOL)saveToPath:(NSString*)path error:(NSError**)error;
- (nullable NSData*)toBytesWithError:(NSError**)error;
/// Number of pages, or -1 on error (sets `error`).
- (NSInteger)pageCountError:(NSError**)error;

/// Free the native handle now (idempotent).
- (void)close;

@end

/// A PDF opened for in-place editing. Owns the native DocumentEditor handle and
/// frees it on -close/-dealloc; status-code C functions surface failures as
/// NSError (POXErrorDomain); is_* queries are exposed as BOOL.
@interface POXDocumentEditor : NSObject

/// Open a PDF for editing from a filesystem path.
+ (nullable instancetype)openEditor:(NSString*)path error:(NSError**)error;
/// Open a PDF for editing from in-memory bytes.
+ (nullable instancetype)openFromBytes:(NSData*)data error:(NSError**)error;

/// Number of pages, or -1 on error (sets `error`).
- (NSInteger)pageCountError:(NSError**)error;
/// PDF version as a POXVersion {major, minor}.
- (POXVersion)version;

/// Whether the editor has pending modifications.
- (BOOL)isModified;
/// The source path of the editor (nil if none / on error).
- (nullable NSString*)sourcePathError:(NSError**)error;

/// Document /Info.Producer.
- (nullable NSString*)producerError:(NSError**)error;
- (BOOL)setProducer:(NSString*)value error:(NSError**)error;
/// Document /Info.CreationDate (raw PDF date string).
- (nullable NSString*)creationDateError:(NSError**)error;
- (BOOL)setCreationDate:(NSString*)date error:(NSError**)error;

/// Page operations (page indices are 0-based).
- (BOOL)deletePage:(NSInteger)page error:(NSError**)error;
- (BOOL)movePageFrom:(NSInteger)from to:(NSInteger)to error:(NSError**)error;

/// Rotation.
- (BOOL)rotatePage:(NSInteger)page byDegrees:(NSInteger)degrees error:(NSError**)error;
- (BOOL)rotateAllPages:(NSInteger)degrees error:(NSError**)error;
- (BOOL)setPageRotation:(NSInteger)page
                degrees:(NSInteger)degrees
                  error:(NSError**)error;
/// Page rotation in degrees, or -1 on error (sets `error`).
- (NSInteger)pageRotation:(NSInteger)page error:(NSError**)error;

/// Crop all pages by margins (left/right/top/bottom, user-space units).
- (BOOL)cropMarginsLeft:(float)left
                  right:(float)right
                    top:(float)top
                 bottom:(float)bottom
                  error:(NSError**)error;

/// Page boxes (returned/accepted as a POXBbox {x, y, width, height}).
- (POXBbox)pageCropBox:(NSInteger)page error:(NSError**)error;
- (BOOL)setPageCropBox:(NSInteger)page box:(POXBbox)box error:(NSError**)error;
- (POXBbox)pageMediaBox:(NSInteger)page error:(NSError**)error;
- (BOOL)setPageMediaBox:(NSInteger)page box:(POXBbox)box error:(NSError**)error;

/// Redaction.
- (BOOL)applyAllRedactions:(NSError**)error;
- (BOOL)applyPageRedactions:(NSInteger)page error:(NSError**)error;
- (BOOL)isPageMarkedForRedaction:(NSInteger)page;
- (BOOL)unmarkPageForRedaction:(NSInteger)page error:(NSError**)error;

/// Erase regions.
- (BOOL)eraseRegion:(NSInteger)page
                  x:(float)x
                  y:(float)y
                  w:(float)w
                  h:(float)h
              error:(NSError**)error;
/// Erase multiple rectangles given as an array of POXBbox values.
- (BOOL)eraseRegions:(NSInteger)page
               rects:(NSArray<NSValue*>*)rects
               error:(NSError**)error;
- (BOOL)clearEraseRegions:(NSInteger)page error:(NSError**)error;

/// Flattening.
- (BOOL)flattenForms:(NSError**)error;
- (BOOL)flattenFormsOnPage:(NSInteger)page error:(NSError**)error;
- (BOOL)flattenAnnotations:(NSInteger)page error:(NSError**)error;
- (BOOL)flattenAllAnnotations:(NSError**)error;
/// Number of warnings from the last form-flattening save, or -1 if no handle.
- (NSInteger)flattenWarningsCount;
- (nullable NSString*)flattenWarning:(NSInteger)index error:(NSError**)error;
- (BOOL)isPageMarkedForFlatten:(NSInteger)page;
- (BOOL)unmarkPageForFlatten:(NSInteger)page error:(NSError**)error;

/// Forms.
- (BOOL)setFormField:(NSString*)name value:(NSString*)value error:(NSError**)error;

/// Merge / conversion / embedding.
- (BOOL)mergeFrom:(NSString*)sourcePath error:(NSError**)error;
- (BOOL)mergeFromBytes:(NSData*)data error:(NSError**)error;
/// Convert to PDF/A in place (0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u).
- (BOOL)convertToPdfA:(NSInteger)level error:(NSError**)error;
- (BOOL)embedFile:(NSString*)name data:(NSData*)data error:(NSError**)error;
/// Extract a subset of 0-based page indices to a new in-memory PDF.
- (nullable NSData*)extractPagesToBytes:(NSArray<NSNumber*>*)pages
                                  error:(NSError**)error;

/// Save.
- (BOOL)saveToPath:(NSString*)path error:(NSError**)error;
- (nullable NSData*)saveToBytesWithError:(NSError**)error;
- (nullable NSData*)saveToBytesCompress:(BOOL)compress
                         garbageCollect:(BOOL)garbageCollect
                              linearize:(BOOL)linearize
                                  error:(NSError**)error;
- (BOOL)saveEncryptedToPath:(NSString*)path
               userPassword:(NSString*)userPassword
              ownerPassword:(NSString*)ownerPassword
                      error:(NSError**)error;
- (nullable NSData*)saveEncryptedToBytesWithUserPassword:(NSString*)userPassword
                                           ownerPassword:(NSString*)ownerPassword
                                                   error:(NSError**)error;

// ── Phase-7: geometric redaction + barcode placement ─────────────────────────

/// Queue a redaction rectangle (corner points x1,y1–x2,y2) filled with the
/// DeviceRGB colour r,g,b (0.0–1.0); page index is 0-based.
- (BOOL)redactionAddPage:(NSInteger)page
                      x1:(double)x1
                      y1:(double)y1
                      x2:(double)x2
                      y2:(double)y2
                       r:(double)r
                       g:(double)g
                       b:(double)b
                   error:(NSError**)error;
/// Number of queued redactions for `page`, or -1 on error.
- (int32_t)redactionCount:(NSInteger)page error:(NSError**)error;
/// Destructively apply all queued redactions; returns the number of glyphs
/// removed, or -1 on error.
- (int32_t)redactionApplyScrubMetadata:(BOOL)scrubMetadata
                                     r:(double)r
                                     g:(double)g
                                     b:(double)b
                                 error:(NSError**)error;
/// Standalone metadata/JS/embedded-file scrub; returns the number of top-level
/// constructs removed, or -1 on error.
- (int32_t)redactionScrubMetadataWithError:(NSError**)error;

/// Place a generated barcode on a page (0-based) at x,y with width×height.
- (BOOL)addBarcode:(POXBarcode*)barcode
              page:(NSInteger)page
                 x:(float)x
                 y:(float)y
             width:(float)width
            height:(float)height
             error:(NSError**)error;

/// Import form data from raw FDF / XFDF bytes into the editor's document.
- (BOOL)importFdfBytes:(NSData*)data error:(NSError**)error;
- (BOOL)importXfdfBytes:(NSData*)data error:(NSError**)error;

/// Free the native handle now (idempotent).
- (void)close;

@end

/// A generated barcode / QR code (Phase-7). Owns the native FfiBarcodeImage
/// handle and frees it on -close/-dealloc.
@interface POXBarcode : NSObject

/// Generate a QR code from `data` with the given error-correction level and
/// pixel size.
+ (nullable instancetype)generateQrCode:(NSString*)data
                        errorCorrection:(int32_t)errorCorrection
                                 sizePx:(int32_t)sizePx
                                  error:(NSError**)error;
/// Generate a 1D/2D barcode of `format` from `data` at the given pixel size.
+ (nullable instancetype)generateBarcode:(NSString*)data
                                  format:(int32_t)format
                                  sizePx:(int32_t)sizePx
                                   error:(NSError**)error;

/// The payload data carried by the barcode.
- (nullable NSString*)dataError:(NSError**)error;
/// The barcode format code, or -1 on error.
- (int32_t)formatError:(NSError**)error;
/// The decode confidence (0.0–1.0).
- (float)confidenceError:(NSError**)error;
/// Encode the barcode as a PNG (the `sizePx` hint is advisory).
- (nullable NSData*)imagePngWithSizePx:(int32_t)sizePx error:(NSError**)error;
/// Encode the barcode as an SVG string (the `sizePx` hint is advisory).
- (nullable NSString*)svgWithSizePx:(int32_t)sizePx error:(NSError**)error;

/// Free the native handle now (idempotent).
- (void)close;

@end

/// An OCR engine backed by detection/recognition/dictionary model files
/// (Phase-7). Owns the native engine handle and frees it on -close/-dealloc.
@interface POXOcrEngine : NSObject

/// Create an OCR engine from model + dictionary file paths.
+ (nullable instancetype)createWithDetModelPath:(NSString*)detModelPath
                                   recModelPath:(NSString*)recModelPath
                                       dictPath:(NSString*)dictPath
                                          error:(NSError**)error;

/// The raw native engine pointer (nil after close), for use by
/// -[POXDocument ocrExtractText:engine:error:].
- (nullable void*)POX_engineHandle;

/// Free the native handle now (idempotent).
- (void)close;

@end

/// A layout-element list for a page (Phase-7). Owns the native FfiElementList
/// handle and frees it on -close/-dealloc. Per-element accessors are exposed via
/// the index-based getters below.
@interface POXElementList : NSObject

/// Number of elements in the list.
- (int32_t)count;
/// The element type string at `index` (nil on error).
- (nullable NSString*)typeAtIndex:(int32_t)index error:(NSError**)error;
/// The element text at `index` (nil on error).
- (nullable NSString*)textAtIndex:(int32_t)index error:(NSError**)error;
/// The element bounding box at `index`.
- (POXBbox)rectAtIndex:(int32_t)index error:(NSError**)error;
/// Serialize the whole list to JSON (nil on error).
- (nullable NSString*)toJsonWithError:(NSError**)error;

/// Free the native handle now (idempotent).
- (void)close;

@end

@class POXDocumentBuilder;
@class POXPageBuilder;

/// A TTF/OTF font loaded for embedding via -[POXDocumentBuilder
/// registerEmbeddedFont:font:error:]. Owns the native EmbeddedFont handle and
/// frees it on -close/-dealloc. NOTE: a successful registerEmbeddedFont:font:
/// transfers ownership of the native handle to the builder — after that the
/// wrapper's handle is nulled out and -close/-dealloc become no-ops.
@interface POXEmbeddedFont : NSObject

/// Load a TTF/OTF font from a filesystem path.
+ (nullable instancetype)fromPath:(NSString*)path error:(NSError**)error;
/// Load a font from in-memory bytes; `name` may be nil to use the PostScript
/// name from the font face.
+ (nullable instancetype)fromBytes:(NSData*)data
                              name:(nullable NSString*)name
                             error:(NSError**)error;

/// Free the native handle now (idempotent; no-op after a successful register).
- (void)close;

@end

/// A page being built within a POXDocumentBuilder. Owns the native
/// FfiPageBuilder handle. The fluent ops return self for chaining; each raises
/// nothing — failures surface via the trailing error/return value. Call -done
/// to commit the page to its parent builder (which consumes the native handle),
/// or -close to discard it. Frees the native handle on -close/-dealloc unless
/// already consumed by -done.
@interface POXPageBuilder : NSObject

// Text layout.
- (BOOL)font:(NSString*)name size:(float)size error:(NSError**)error;
- (BOOL)at:(float)x y:(float)y error:(NSError**)error;
- (BOOL)text:(NSString*)text error:(NSError**)error;
- (BOOL)heading:(uint8_t)level text:(NSString*)text error:(NSError**)error;
- (BOOL)paragraph:(NSString*)text error:(NSError**)error;
- (BOOL)space:(float)points error:(NSError**)error;
- (BOOL)horizontalRule:(NSError**)error;

// Links.
- (BOOL)linkUrl:(NSString*)url error:(NSError**)error;
- (BOOL)linkPage:(NSInteger)page error:(NSError**)error;
- (BOOL)linkNamed:(NSString*)destination error:(NSError**)error;
- (BOOL)linkJavascript:(NSString*)script error:(NSError**)error;

// Page-level actions.
- (BOOL)onOpen:(NSString*)script error:(NSError**)error;
- (BOOL)onClose:(NSString*)script error:(NSError**)error;

// Form-field JS actions (apply to the most-recently-added field).
- (BOOL)fieldKeystroke:(NSString*)script error:(NSError**)error;
- (BOOL)fieldFormat:(NSString*)script error:(NSError**)error;
- (BOOL)fieldValidate:(NSString*)script error:(NSError**)error;
- (BOOL)fieldCalculate:(NSString*)script error:(NSError**)error;

// Text decorations (RGB channels 0.0–1.0; apply to the previous text element).
- (BOOL)highlightR:(float)r g:(float)g b:(float)b error:(NSError**)error;
- (BOOL)underlineR:(float)r g:(float)g b:(float)b error:(NSError**)error;
- (BOOL)strikeoutR:(float)r g:(float)g b:(float)b error:(NSError**)error;
- (BOOL)squigglyR:(float)r g:(float)g b:(float)b error:(NSError**)error;

// Annotations.
- (BOOL)stickyNote:(NSString*)text error:(NSError**)error;
- (BOOL)stickyNoteAt:(float)x y:(float)y text:(NSString*)text error:(NSError**)error;
- (BOOL)watermark:(NSString*)text error:(NSError**)error;
- (BOOL)watermarkConfidential:(NSError**)error;
- (BOOL)watermarkDraft:(NSError**)error;
- (BOOL)stamp:(NSString*)typeName error:(NSError**)error;
- (BOOL)freetextAtX:(float)x
                  y:(float)y
                  w:(float)w
                  h:(float)h
               text:(NSString*)text
              error:(NSError**)error;

// Form fields.
- (BOOL)textFieldName:(NSString*)name
                    x:(float)x
                    y:(float)y
                    w:(float)w
                    h:(float)h
         defaultValue:(nullable NSString*)defaultValue
                error:(NSError**)error;
- (BOOL)checkboxName:(NSString*)name
                   x:(float)x
                   y:(float)y
                   w:(float)w
                   h:(float)h
             checked:(BOOL)checked
               error:(NSError**)error;
- (BOOL)comboBoxName:(NSString*)name
                   x:(float)x
                   y:(float)y
                   w:(float)w
                   h:(float)h
             options:(NSArray<NSString*>*)options
            selected:(nullable NSString*)selected
               error:(NSError**)error;
- (BOOL)radioGroupName:(NSString*)name
                values:(NSArray<NSString*>*)values
                    xs:(NSArray<NSNumber*>*)xs
                    ys:(NSArray<NSNumber*>*)ys
                    ws:(NSArray<NSNumber*>*)ws
                    hs:(NSArray<NSNumber*>*)hs
              selected:(nullable NSString*)selected
                 error:(NSError**)error;
- (BOOL)pushButtonName:(NSString*)name
                     x:(float)x
                     y:(float)y
                     w:(float)w
                     h:(float)h
               caption:(NSString*)caption
                 error:(NSError**)error;
- (BOOL)signatureFieldName:(NSString*)name
                         x:(float)x
                         y:(float)y
                         w:(float)w
                         h:(float)h
                     error:(NSError**)error;

// Footnotes / columns / inline runs.
- (BOOL)footnoteRefMark:(NSString*)refMark
               noteText:(NSString*)noteText
                  error:(NSError**)error;
- (BOOL)columnsCount:(uint32_t)columnCount
               gapPt:(float)gapPt
                text:(NSString*)text
               error:(NSError**)error;
- (BOOL)inlineText:(NSString*)text error:(NSError**)error;
- (BOOL)inlineBold:(NSString*)text error:(NSError**)error;
- (BOOL)inlineItalic:(NSString*)text error:(NSError**)error;
- (BOOL)inlineColorR:(float)r
                   g:(float)g
                   b:(float)b
                text:(NSString*)text
               error:(NSError**)error;
- (BOOL)newline:(NSError**)error;

// Barcodes.
- (BOOL)barcode1d:(int32_t)barcodeType
             data:(NSString*)data
                x:(float)x
                y:(float)y
                w:(float)w
                h:(float)h
            error:(NSError**)error;
- (BOOL)barcodeQrData:(NSString*)data
                    x:(float)x
                    y:(float)y
                 size:(float)size
                error:(NSError**)error;

// Images.
- (BOOL)image:(NSData*)bytes
            x:(float)x
            y:(float)y
            w:(float)w
            h:(float)h
        error:(NSError**)error;
- (BOOL)imageWithAlt:(NSData*)bytes
                   x:(float)x
                   y:(float)y
                   w:(float)w
                   h:(float)h
             altText:(NSString*)altText
               error:(NSError**)error;
- (BOOL)imageArtifact:(NSData*)bytes
                    x:(float)x
                    y:(float)y
                    w:(float)w
                    h:(float)h
                error:(NSError**)error;

// Vector graphics.
- (BOOL)rectX:(float)x y:(float)y w:(float)w h:(float)h error:(NSError**)error;
- (BOOL)filledRectX:(float)x
                  y:(float)y
                  w:(float)w
                  h:(float)h
                  r:(float)r
                  g:(float)g
                  b:(float)b
              error:(NSError**)error;
- (BOOL)lineX1:(float)x1 y1:(float)y1 x2:(float)x2 y2:(float)y2 error:(NSError**)error;
- (BOOL)strokeRectX:(float)x
                  y:(float)y
                  w:(float)w
                  h:(float)h
              width:(float)width
                  r:(float)r
                  g:(float)g
                  b:(float)b
              error:(NSError**)error;
- (BOOL)strokeLineX1:(float)x1
                  y1:(float)y1
                  x2:(float)x2
                  y2:(float)y2
               width:(float)width
                   r:(float)r
                   g:(float)g
                   b:(float)b
               error:(NSError**)error;
- (BOOL)strokeRectDashedX:(float)x
                        y:(float)y
                        w:(float)w
                        h:(float)h
                    width:(float)width
                        r:(float)r
                        g:(float)g
                        b:(float)b
                dashArray:(NSArray<NSNumber*>*)dashArray
                    phase:(float)phase
                    error:(NSError**)error;
- (BOOL)strokeLineDashedX1:(float)x1
                        y1:(float)y1
                        x2:(float)x2
                        y2:(float)y2
                     width:(float)width
                         r:(float)r
                         g:(float)g
                         b:(float)b
                 dashArray:(NSArray<NSNumber*>*)dashArray
                     phase:(float)phase
                     error:(NSError**)error;

// Misc layout.
- (BOOL)textInRectX:(float)x
                  y:(float)y
                  w:(float)w
                  h:(float)h
               text:(NSString*)text
              align:(int32_t)align
              error:(NSError**)error;
- (BOOL)newPageSameSize:(NSError**)error;

// Tables.
- (BOOL)tableColumns:(NSArray<NSNumber*>*)widths
              aligns:(NSArray<NSNumber*>*)aligns
                rows:(NSArray<NSArray<NSString*>*>*)rows
           hasHeader:(BOOL)hasHeader
               error:(NSError**)error;

// Streaming tables.
- (BOOL)streamingTableBeginHeaders:(NSArray<NSString*>*)headers
                            widths:(NSArray<NSNumber*>*)widths
                            aligns:(NSArray<NSNumber*>*)aligns
                      repeatHeader:(BOOL)repeatHeader
                             error:(NSError**)error;
- (BOOL)streamingTableBeginV2Headers:(NSArray<NSString*>*)headers
                              widths:(NSArray<NSNumber*>*)widths
                              aligns:(NSArray<NSNumber*>*)aligns
                        repeatHeader:(BOOL)repeatHeader
                                mode:(int32_t)mode
                          sampleRows:(NSInteger)sampleRows
                       minColWidthPt:(float)minColWidthPt
                       maxColWidthPt:(float)maxColWidthPt
                          maxRowspan:(NSInteger)maxRowspan
                               error:(NSError**)error;
- (BOOL)streamingTableSetBatchSize:(NSInteger)batchSize error:(NSError**)error;
- (NSInteger)streamingTablePendingRowCount;
- (NSInteger)streamingTableBatchCount;
- (BOOL)streamingTableFlush:(NSError**)error;
- (BOOL)streamingTablePushRow:(NSArray<NSString*>*)cells error:(NSError**)error;
- (BOOL)streamingTablePushRowV2:(NSArray<NSString*>*)cells
                       rowspans:(nullable NSArray<NSNumber*>*)rowspans
                          error:(NSError**)error;
- (BOOL)streamingTableFinish:(NSError**)error;

/// Commit this page to its parent builder and consume the native handle.
- (BOOL)done:(NSError**)error;
/// Discard this page without committing (idempotent).
- (void)close;

@end

/// A fluent builder that constructs a brand-new PDF. Owns the native
/// FfiDocumentBuilder handle and frees it on -close/-dealloc. Metadata setters
/// return YES on success; -page:/-letterPage/-a4Page return a POXPageBuilder
/// bound to this builder; -build/-toBytesEncrypted return the PDF bytes.
@interface POXDocumentBuilder : NSObject

/// Create a new, empty document builder.
+ (nullable instancetype)createWithError:(NSError**)error;

// Metadata.
- (BOOL)setTitle:(NSString*)title error:(NSError**)error;
- (BOOL)setAuthor:(NSString*)author error:(NSError**)error;
- (BOOL)setSubject:(NSString*)subject error:(NSError**)error;
- (BOOL)setKeywords:(NSString*)keywords error:(NSError**)error;
- (BOOL)setCreator:(NSString*)creator error:(NSError**)error;
- (BOOL)onOpen:(NSString*)script error:(NSError**)error;

// Tagged / accessible PDF.
- (BOOL)taggedPdfUa1:(NSError**)error;
- (BOOL)language:(NSString*)lang error:(NSError**)error;
- (BOOL)roleMapCustom:(NSString*)custom
             standard:(NSString*)standard
                error:(NSError**)error;

/// Register a TTF/OTF font under `name`. On success the builder takes ownership
/// of the font's native handle (the POXEmbeddedFont becomes inert).
- (BOOL)registerEmbeddedFont:(NSString*)name
                        font:(POXEmbeddedFont*)font
                       error:(NSError**)error;

// Pages.
- (nullable POXPageBuilder*)a4PageWithError:(NSError**)error;
- (nullable POXPageBuilder*)letterPageWithError:(NSError**)error;
- (nullable POXPageBuilder*)pageWithWidth:(float)width
                                   height:(float)height
                                    error:(NSError**)error;

// Build / save.
- (nullable NSData*)buildWithError:(NSError**)error;
- (BOOL)saveToPath:(NSString*)path error:(NSError**)error;
- (BOOL)saveEncryptedToPath:(NSString*)path
               userPassword:(NSString*)userPassword
              ownerPassword:(NSString*)ownerPassword
                      error:(NSError**)error;
- (nullable NSData*)toBytesEncryptedWithUserPassword:(NSString*)userPassword
                                       ownerPassword:(NSString*)ownerPassword
                                               error:(NSError**)error;

/// Free the native handle now (idempotent).
- (void)close;

@end

// ── Phase-6: digital signatures / PKI / timestamps / conformance ─────────────

/// PDF/UA accessibility element counts (from -[POXUaResults stats]).
typedef struct {
    int32_t structElements;
    int32_t images;
    int32_t tables;
    int32_t forms;
    int32_t annotations;
    int32_t pages;
} POXUaStats;

/// A loaded signing certificate / verifier certificate. Owns the native
/// Certificate handle and frees it on -close/-dealloc.
@interface POXCertificate : NSObject

/// Load signing credentials from PKCS#12 (.p12/.pfx) bytes + password.
+ (nullable instancetype)loadFromBytes:(NSData*)data
                              password:(NSString*)password
                                 error:(NSError**)error;
/// Load signing credentials from PEM-encoded certificate + private key.
+ (nullable instancetype)loadFromPemCert:(NSString*)certPem
                                  keyPem:(NSString*)keyPem
                                   error:(NSError**)error;

- (nullable NSString*)subjectError:(NSError**)error;
- (nullable NSString*)issuerError:(NSError**)error;
- (nullable NSString*)serialError:(NSError**)error;
/// Validity window as Unix epoch seconds; returns NO on error (sets `error`).
- (BOOL)validityNotBefore:(int64_t*)notBefore
                 notAfter:(int64_t*)notAfter
                    error:(NSError**)error;
/// 1 if currently valid, 0 if not, -1 on error.
- (int32_t)isValidError:(NSError**)error;

/// Free the native handle now (idempotent).
- (void)close;

@end

/// Information about a signature read from a document. Owns the native
/// FfiSignatureInfo handle and frees it on -close/-dealloc.
@interface POXSignatureInfo : NSObject

- (nullable NSString*)signerNameError:(NSError**)error;
- (nullable NSString*)signingReasonError:(NSError**)error;
- (nullable NSString*)signingLocationError:(NSError**)error;
/// Signing time as Unix epoch seconds.
- (int64_t)signingTimeError:(NSError**)error;
/// The signer certificate (nil on error / absent).
- (nullable POXCertificate*)certificateError:(NSError**)error;
/// PAdES baseline level (0=B-B 1=B-T 2=B-LT …), or -1 on error.
- (int32_t)padesLevelError:(NSError**)error;
- (BOOL)hasTimestampError:(NSError**)error;
/// The embedded signature timestamp (nil on error / absent).
- (nullable POXTimestamp*)timestampError:(NSError**)error;
/// Attach a timestamp to this signature; returns YES on success.
- (BOOL)addTimestamp:(POXTimestamp*)timestamp error:(NSError**)error;
/// Signer-attribute crypto check: 1 valid, 0 invalid, -1 unknown/unsupported.
- (int32_t)verifyError:(NSError**)error;
/// End-to-end verify against the full PDF bytes: 1 valid, 0 invalid, -1 unknown.
- (int32_t)verifyDetached:(NSData*)pdf error:(NSError**)error;

/// Free the native handle now (idempotent).
- (void)close;

@end

/// A parsed RFC 3161 timestamp token. Owns the native Timestamp handle and frees
/// it on -close/-dealloc.
@interface POXTimestamp : NSObject

/// Parse a DER-encoded TimeStampToken (or bare TSTInfo).
+ (nullable instancetype)parse:(NSData*)data error:(NSError**)error;

/// The raw DER token bytes (copied; nil on error).
- (nullable NSData*)tokenError:(NSError**)error;
/// The message imprint bytes (copied; nil on error).
- (nullable NSData*)messageImprintError:(NSError**)error;
/// Timestamp time as Unix epoch seconds.
- (int64_t)timeError:(NSError**)error;
- (nullable NSString*)serialError:(NSError**)error;
- (nullable NSString*)tsaNameError:(NSError**)error;
- (nullable NSString*)policyOidError:(NSError**)error;
/// Hash algorithm code, or -1 on error.
- (int32_t)hashAlgorithmError:(NSError**)error;
- (BOOL)verifyError:(NSError**)error;

/// Free the native handle now (idempotent).
- (void)close;

@end

/// A TSA (RFC 3161) client. Owns the native TsaClient handle and frees it on
/// -close/-dealloc.
@interface POXTsaClient : NSObject

+ (nullable instancetype)createWithUrl:(NSString*)url
                              username:(nullable NSString*)username
                              password:(nullable NSString*)password
                               timeout:(int32_t)timeout
                              hashAlgo:(int32_t)hashAlgo
                              useNonce:(BOOL)useNonce
                               certReq:(BOOL)certReq
                                 error:(NSError**)error;

/// Request a timestamp over the given data (the TSA hashes it).
- (nullable POXTimestamp*)requestTimestamp:(NSData*)data error:(NSError**)error;
/// Request a timestamp over a pre-computed hash.
- (nullable POXTimestamp*)requestTimestampHash:(NSData*)hash
                                      hashAlgo:(int32_t)hashAlgo
                                         error:(NSError**)error;

/// Free the native handle now (idempotent).
- (void)close;

@end

/// A Document Security Store (DSS) read from a document. Owns the native DSS
/// handle and frees it on -close/-dealloc.
@interface POXDss : NSObject

- (int32_t)certCount;
- (int32_t)crlCount;
- (int32_t)ocspCount;
- (int32_t)vriCount;
- (nullable NSData*)certAtIndex:(int32_t)index error:(NSError**)error;
- (nullable NSData*)crlAtIndex:(int32_t)index error:(NSError**)error;
- (nullable NSData*)ocspAtIndex:(int32_t)index error:(NSError**)error;

/// Free the native handle now (idempotent).
- (void)close;

@end

/// PDF/A validation result. Owns the native FfiPdfAResults handle and frees it
/// on -close/-dealloc.
@interface POXPdfAResults : NSObject
- (BOOL)isCompliantError:(NSError**)error;
- (int32_t)errorCount;
- (int32_t)warningCount;
/// All error messages (empty array if none).
- (NSArray<NSString*>*)errors;
/// Free the native handle now (idempotent).
- (void)close;
@end

/// PDF/UA accessibility validation result. Owns the native FfiUaResults handle
/// and frees it on -close/-dealloc.
@interface POXUaResults : NSObject
- (BOOL)isAccessibleError:(NSError**)error;
- (int32_t)errorCount;
- (int32_t)warningCount;
- (NSArray<NSString*>*)errors;
- (NSArray<NSString*>*)warnings;
/// Accessibility element counts; returns NO on error (sets `error`).
- (BOOL)stats:(POXUaStats*)stats error:(NSError**)error;
/// Free the native handle now (idempotent).
- (void)close;
@end

/// PDF/X validation result. Owns the native FfiPdfXResults handle and frees it
/// on -close/-dealloc.
@interface POXPdfXResults : NSObject
- (BOOL)isCompliantError:(NSError**)error;
- (int32_t)errorCount;
- (NSArray<NSString*>*)errors;
/// Free the native handle now (idempotent).
- (void)close;
@end

/// PAdES signing options (struct-pointer variant of PAdES signing). The
/// revocation arrays carry DER-encoded certs/CRLs/OCSP responses (B-LT material).
@interface POXPadesSignOptions : NSObject
@property(nonatomic, strong) POXCertificate* certificate;
@property(nonatomic) int32_t level; ///< 0=B-B 1=B-T 2=B-LT
@property(nonatomic, copy, nullable) NSString* tsaUrl;
@property(nonatomic, copy, nullable) NSString* reason;
@property(nonatomic, copy, nullable) NSString* location;
@property(nonatomic, copy) NSArray<NSData*>* certs;
@property(nonatomic, copy) NSArray<NSData*>* crls;
@property(nonatomic, copy) NSArray<NSData*>* ocsps;
@end

/// Top-level signing + library-configuration entry points (Phase-6).
@interface POXSigning : NSObject

/// Sign raw PDF bytes with a basic (non-PAdES) signature.
+ (nullable NSData*)signBytes:(NSData*)pdf
                  certificate:(POXCertificate*)certificate
                       reason:(nullable NSString*)reason
                     location:(nullable NSString*)location
                        error:(NSError**)error;

/// Sign raw PDF bytes at a PAdES baseline level.
+ (nullable NSData*)signBytesPades:(NSData*)pdf
                       certificate:(POXCertificate*)certificate
                             level:(int32_t)level
                            tsaUrl:(nullable NSString*)tsaUrl
                            reason:(nullable NSString*)reason
                          location:(nullable NSString*)location
                             certs:(NSArray<NSData*>*)certs
                              crls:(NSArray<NSData*>*)crls
                             ocsps:(NSArray<NSData*>*)ocsps
                             error:(NSError**)error;

/// Sign raw PDF bytes using a POXPadesSignOptions config (struct variant).
+ (nullable NSData*)signBytesPadesOpts:(NSData*)pdf
                               options:(POXPadesSignOptions*)options
                                 error:(NSError**)error;

/// Set the global log level (0=Off 1=Error 2=Warn 3=Info 4=Debug 5=Trace).
+ (void)setLogLevel:(int32_t)level;
/// Get the current global log level (0-5).
+ (int32_t)logLevel;

@end

// ── Phase-7: top-level merge + timestamp helpers ─────────────────────────────

/// Top-level Phase-7 utility entry points that are not bound to a handle.
@interface POXTools : NSObject

/// Merge the PDFs at `paths` (in order) into a single in-memory PDF.
+ (nullable NSData*)merge:(NSArray<NSString*>*)paths error:(NSError**)error;

/// Append an RFC 3161 document timestamp to the signature at `sigIndex` in the
/// given PDF bytes, fetching the token from `tsaUrl`. On success returns the
/// timestamped PDF bytes; nil (with `error`) on failure.
+ (nullable NSData*)addTimestamp:(NSData*)pdfData
                        sigIndex:(int32_t)sigIndex
                          tsaUrl:(NSString*)tsaUrl
                           error:(NSError**)error;

@end

/// Crypto-provider introspection and FIPS policy (process-global).
@interface POXCrypto : NSObject
/// Name of the active crypto provider (nil on error).
+ (nullable NSString*)activeProviderWithError:(NSError**)error;
/// CycloneDX CBOM JSON for the active provider (nil on error).
+ (nullable NSString*)cbomWithError:(NSError**)error;
/// Crypto inventory JSON (nil on error).
+ (nullable NSString*)inventoryWithError:(NSError**)error;
/// Active crypto policy JSON / descriptor (nil on error).
+ (nullable NSString*)policyWithError:(NSError**)error;
/// 1 if a FIPS-validated provider is available, 0 if not, -1 on error.
+ (int32_t)fipsAvailable;
/// Switch to the FIPS provider; returns 0 on success.
+ (int32_t)useFips;
/// Set the crypto policy from a spec string; returns 0 on success.
+ (int32_t)setPolicy:(NSString*)spec;
@end

/// Model manifest / prefetch (offline ML model management).
@interface POXModels : NSObject
/// Bundled model manifest JSON (nil on error).
+ (nullable NSString*)manifestWithError:(NSError**)error;
/// 1 if model prefetch is available, 0 if not, -1 on error.
+ (int32_t)prefetchAvailable;
/// Prefetch models for the comma-separated language codes; returns a status JSON
/// (nil on error — e.g. when network/models are unavailable).
+ (nullable NSString*)prefetchModels:(NSString*)languagesCsv error:(NSError**)error;
@end

/// Process-global configuration knobs and the standalone renderer handle.
@interface POXConfig : NSObject
/// Cap operators decoded per content stream; returns the previous limit.
+ (int64_t)setMaxOpsPerStream:(int64_t)limit;
/// Toggle keeping unmapped glyphs in extracted text; returns the previous flag.
+ (int32_t)setPreserveUnmappedGlyphs:(int32_t)preserve;
@end

/// A standalone renderer handle (independent of a document). Owns the native
/// handle and frees it on -close/-dealloc.
@interface POXRenderer : NSObject
+ (nullable instancetype)createWithDpi:(int32_t)dpi
                                format:(int32_t)format
                               quality:(int32_t)quality
                             antiAlias:(BOOL)antiAlias
                                 error:(NSError**)error;
/// Free the native handle now (idempotent).
- (void)close;
@end

NS_ASSUME_NONNULL_END
