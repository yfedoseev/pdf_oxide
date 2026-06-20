// pdf_oxide — Objective-C binding implementation (over the C ABI).
#import "POXPdfOxide.h"
#import <pdf_oxide_c/pdf_oxide.h>

NSString *const POXErrorDomain = @"fyi.oxide.pdf";

static NSError *POXMakeError(int32_t code, NSString *op) {
    return [NSError errorWithDomain:POXErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey :
                                          [NSString stringWithFormat:@"pdf_oxide: %@ failed (error code %d)", op, code]}];
}

// Copy a C string return into NSString and free it via free_string.
static NSString *_Nullable POXTakeString(char *s, int32_t code, NSString *op, NSError **error) {
    if (s == NULL) {
        if (error) *error = POXMakeError(code, op);
        return nil;
    }
    NSString *out = [NSString stringWithUTF8String:s];
    free_string(s);
    return out;
}

@implementation POXDocument {
    PdfDocument *_handle;
}

+ (instancetype)openPath:(NSString *)path error:(NSError **)error {
    int32_t code = 0;
    PdfDocument *h = pdf_document_open(path.UTF8String, &code);
    if (!h) { if (error) *error = POXMakeError(code, @"open"); return nil; }
    return [[self alloc] initWithHandle:h];
}

+ (instancetype)openData:(NSData *)data error:(NSError **)error {
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_from_bytes(data.bytes, data.length, &code);
    if (!h) { if (error) *error = POXMakeError(code, @"openData"); return nil; }
    return [[self alloc] initWithHandle:h];
}

+ (instancetype)openPath:(NSString *)path password:(NSString *)password error:(NSError **)error {
    int32_t code = 0;
    PdfDocument *h = pdf_document_open_with_password(path.UTF8String, password.UTF8String, &code);
    if (!h) { if (error) *error = POXMakeError(code, @"openPassword"); return nil; }
    return [[self alloc] initWithHandle:h];
}

- (instancetype)initWithHandle:(PdfDocument *)handle {
    if ((self = [super init])) { _handle = handle; }
    return self;
}

- (void)dealloc {
    if (_handle) pdf_document_free(_handle);
}

- (NSInteger)pageCountError:(NSError **)error {
    int32_t code = 0;
    int32_t n = pdf_document_get_page_count(_handle, &code);
    if (n < 0) { if (error) *error = POXMakeError(code, @"pageCount"); return -1; }
    return n;
}

- (void)getVersionMajor:(uint8_t *)major minor:(uint8_t *)minor {
    pdf_document_get_version(_handle, major, minor);
}

- (BOOL)isEncrypted { return pdf_document_is_encrypted(_handle); }
- (BOOL)hasStructureTree { return pdf_document_has_structure_tree(_handle); }

- (NSString *)extractText:(NSInteger)page error:(NSError **)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_extract_text(_handle, (int32_t)page, &code), code, @"extractText", error);
}
- (NSString *)toPlainText:(NSInteger)page error:(NSError **)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_to_plain_text(_handle, (int32_t)page, &code), code, @"toPlainText", error);
}
- (NSString *)toMarkdown:(NSInteger)page error:(NSError **)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_to_markdown(_handle, (int32_t)page, &code), code, @"toMarkdown", error);
}
- (NSString *)toHtml:(NSInteger)page error:(NSError **)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_to_html(_handle, (int32_t)page, &code), code, @"toHtml", error);
}
- (NSString *)toMarkdownAllError:(NSError **)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_to_markdown_all(_handle, &code), code, @"toMarkdownAll", error);
}
- (NSString *)extractStructuredJson:(NSInteger)page error:(NSError **)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_extract_structured_to_json(_handle, (int32_t)page, &code), code, @"extractStructuredJson", error);
}

@end

@implementation POXPdf {
    Pdf *_handle;
}

+ (instancetype)fromMarkdown:(NSString *)markdown error:(NSError **)error {
    int32_t code = 0;
    Pdf *h = pdf_from_markdown(markdown.UTF8String, &code);
    if (!h) { if (error) *error = POXMakeError(code, @"fromMarkdown"); return nil; }
    return [[self alloc] initWithHandle:h];
}
+ (instancetype)fromHtml:(NSString *)html error:(NSError **)error {
    int32_t code = 0;
    Pdf *h = pdf_from_html(html.UTF8String, &code);
    if (!h) { if (error) *error = POXMakeError(code, @"fromHtml"); return nil; }
    return [[self alloc] initWithHandle:h];
}
+ (instancetype)fromText:(NSString *)text error:(NSError **)error {
    int32_t code = 0;
    Pdf *h = pdf_from_text(text.UTF8String, &code);
    if (!h) { if (error) *error = POXMakeError(code, @"fromText"); return nil; }
    return [[self alloc] initWithHandle:h];
}

- (instancetype)initWithHandle:(Pdf *)handle {
    if ((self = [super init])) { _handle = handle; }
    return self;
}

- (void)dealloc {
    if (_handle) pdf_free(_handle);
}

- (BOOL)saveToPath:(NSString *)path error:(NSError **)error {
    int32_t code = 0;
    if (pdf_save(_handle, path.UTF8String, &code) != 0) {
        if (error) *error = POXMakeError(code, @"save");
        return NO;
    }
    return YES;
}

- (NSData *)saveToBytesError:(NSError **)error {
    int32_t len = 0, code = 0;
    uint8_t *p = pdf_save_to_bytes(_handle, &len, &code);
    if (!p) { if (error) *error = POXMakeError(code, @"saveToBytes"); return nil; }
    NSData *out = [NSData dataWithBytes:p length:(len < 0 ? 0 : (NSUInteger)len)];
    free_string((char *)p);
    return out;
}

@end
