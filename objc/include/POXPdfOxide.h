// pdf_oxide — idiomatic Objective-C bindings over the C ABI.
//
// NSObject wrappers (POXDocument, POXPdf) own the C handles and free them in
// -dealloc; returned C strings/buffers are copied into NSString/NSData and
// freed via free_string; non-success C-ABI error codes surface as NSError
// (POXErrorDomain). API surface mirrors the other language bindings; coverage
// is asserted by POXApiCoverageTests (one test per method).
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

extern NSString *const POXErrorDomain;

/// An opened PDF for extraction/inspection.
@interface POXDocument : NSObject

/// Open a PDF from a filesystem path.
+ (nullable instancetype)openPath:(NSString *)path error:(NSError **)error;
/// Open a PDF from in-memory bytes.
+ (nullable instancetype)openData:(NSData *)data error:(NSError **)error;
/// Open a password-protected PDF.
+ (nullable instancetype)openPath:(NSString *)path
                         password:(NSString *)password
                            error:(NSError **)error;

/// Number of pages, or -1 on error (sets `error`).
- (NSInteger)pageCountError:(NSError **)error;
/// PDF version major/minor (out params).
- (void)getVersionMajor:(uint8_t *)major minor:(uint8_t *)minor;
- (BOOL)isEncrypted;
- (BOOL)hasStructureTree;

- (nullable NSString *)extractText:(NSInteger)page error:(NSError **)error;
- (nullable NSString *)toPlainText:(NSInteger)page error:(NSError **)error;
- (nullable NSString *)toMarkdown:(NSInteger)page error:(NSError **)error;
- (nullable NSString *)toHtml:(NSInteger)page error:(NSError **)error;
- (nullable NSString *)toMarkdownAllError:(NSError **)error;
- (nullable NSString *)extractStructuredJson:(NSInteger)page error:(NSError **)error;

@end

/// A PDF produced by a builder.
@interface POXPdf : NSObject

+ (nullable instancetype)fromMarkdown:(NSString *)markdown error:(NSError **)error;
+ (nullable instancetype)fromHtml:(NSString *)html error:(NSError **)error;
+ (nullable instancetype)fromText:(NSString *)text error:(NSError **)error;

- (BOOL)saveToPath:(NSString *)path error:(NSError **)error;
- (nullable NSData *)saveToBytesError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
