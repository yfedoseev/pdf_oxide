// pdf_oxide — idiomatic Objective-C bindings over the C ABI.
//
// NSObject wrappers (POXDocument, POXPdf) own the C handles and free them in
// -dealloc; returned C strings/buffers are copied into NSString/NSData and
// freed via free_string; non-success C-ABI error codes surface as NSError
// (POXErrorDomain). API surface mirrors the other language bindings; coverage
// is asserted by POXApiCoverageTests (one test per method).
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

extern NSString* const POXErrorDomain;

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

- (BOOL)saveToPath:(NSString*)path error:(NSError**)error;
- (nullable NSData*)toBytesWithError:(NSError**)error;

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

/// Free the native handle now (idempotent).
- (void)close;

@end

NS_ASSUME_NONNULL_END
