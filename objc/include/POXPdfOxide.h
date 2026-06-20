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

NS_ASSUME_NONNULL_END
