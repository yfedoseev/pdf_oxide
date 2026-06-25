// pdf_oxide — Objective-C binding implementation (over the C ABI).
#import "POXPdfOxide.h"
#import <pdf_oxide_c/pdf_oxide.h>

NSString* const POXErrorDomain = @"fyi.oxide.pdf";

static NSError* POXMakeError(int32_t code, NSString* op) {
    return [NSError
        errorWithDomain:POXErrorDomain
                   code:code
               userInfo:@{
                   NSLocalizedDescriptionKey : [NSString
                       stringWithFormat:@"pdf_oxide: %@ failed (error code %d)", op,
                                        code]
               }];
}

// Copy an owned byte buffer into NSData and free it via free_bytes (defined
// later); forward-declared so document-level byte returns can use it.
static NSData* _Nullable POXTakeBytes(uint8_t* p, NSUInteger len, int32_t code,
                                      NSString* op, NSError** error);

// Copy a C string return into NSString and free it via free_string.
static NSString* _Nullable POXTakeString(char* s, int32_t code, NSString* op,
                                         NSError** error) {
    if (s == NULL) {
        if (error)
            *error = POXMakeError(code, op);
        return nil;
    }
    NSString* out = [NSString stringWithUTF8String:s];
    free_string(s);
    return out;
}

// Private initializer used by -[POXDocument pageAtIndex:].
@interface POXPage ()
- (instancetype)initWithDocument:(POXDocument*)document index:(NSInteger)index;
@end

// Phase-6 private handle-taking initializers (used before the @implementation).
// `POX_handle` exposes the raw native pointer to sibling Phase-6 types in this
// translation unit (signing helpers, addTimestamp).
@interface POXCertificate ()
- (instancetype)initWithHandle:(void*)handle;
- (void*)POX_handle;
@end
@interface POXTimestamp ()
- (instancetype)initWithHandle:(void*)handle;
- (void*)POX_handle;
@end
@interface POXSignatureInfo ()
- (instancetype)initWithHandle:(FfiSignatureInfo*)handle;
@end
@interface POXDss ()
- (instancetype)initWithHandle:(void*)handle;
@end
@interface POXPdfAResults ()
- (instancetype)initWithHandle:(FfiPdfAResults*)handle;
@end
@interface POXUaResults ()
- (instancetype)initWithHandle:(FfiUaResults*)handle;
@end
@interface POXPdfXResults ()
- (instancetype)initWithHandle:(FfiPdfXResults*)handle;
@end

// Phase-7 private handle-taking initializers / handle accessors.
@interface POXBarcode ()
- (instancetype)initWithHandle:(FfiBarcodeImage*)handle;
- (const FfiBarcodeImage*)POX_handle;
@end
@interface POXElementList ()
- (instancetype)initWithHandle:(FfiElementList*)handle;
@end

// ── Phase-1 element model types ──────────────────────────────────────────────

@interface POXChar ()
- (instancetype)initWithCharacter:(uint32_t)character
                             bbox:(POXBbox)bbox
                         fontName:(NSString*)fontName
                         fontSize:(float)fontSize;
@end

@interface POXWord ()
- (instancetype)initWithText:(NSString*)text
                        bbox:(POXBbox)bbox
                    fontName:(NSString*)fontName
                    fontSize:(float)fontSize
                        bold:(BOOL)bold;
@end

@interface POXTextLine ()
- (instancetype)initWithText:(NSString*)text
                        bbox:(POXBbox)bbox
                   wordCount:(NSInteger)wordCount;
@end

@interface POXTable ()
- (instancetype)initWithRowCount:(NSInteger)rowCount
                        colCount:(NSInteger)colCount
                       hasHeader:(BOOL)hasHeader
                           cells:(NSArray<NSArray<NSString*>*>*)cells;
@end

// ── Phase-2 element model types ──────────────────────────────────────────────

@interface POXFont ()
- (instancetype)initWithName:(NSString*)name
                        type:(NSString*)type
                    encoding:(NSString*)encoding
                    embedded:(BOOL)embedded
                      subset:(BOOL)subset;
@property(nonatomic, readwrite) float size;
@end

@interface POXImage ()
- (instancetype)initWithWidth:(NSInteger)width
                       height:(NSInteger)height
             bitsPerComponent:(NSInteger)bitsPerComponent
                       format:(NSString*)format
                   colorspace:(NSString*)colorspace
                         data:(NSData*)data;
@end

@interface POXAnnotation ()
- (instancetype)initWithType:(NSString*)type
                     subtype:(NSString*)subtype
                     content:(NSString*)content
                      author:(NSString*)author
                        rect:(POXBbox)rect
                 borderWidth:(float)borderWidth;
@property(nonatomic, readwrite) uint32_t color;
@property(nonatomic, readwrite) int64_t creationDate;
@property(nonatomic, readwrite) int64_t modificationDate;
@property(nonatomic, readwrite) BOOL hidden;
@property(nonatomic, readwrite) BOOL markedDeleted;
@property(nonatomic, readwrite) BOOL printable;
@property(nonatomic, readwrite) BOOL readOnly;
@property(nonatomic, readwrite, copy, nullable) NSString* linkUri;
@property(nonatomic, readwrite, copy, nullable) NSString* iconName;
@property(nonatomic, readwrite, copy) NSArray<NSArray<NSNumber*>*>* quadPoints;
@end

@interface POXFormField ()
- (instancetype)initWithName:(NSString*)name
                       value:(NSString*)value
                        type:(NSString*)type
                    readonly:(BOOL)readonly
                    required:(BOOL)required;
@end

@interface POXPath ()
- (instancetype)initWithBbox:(POXBbox)bbox
                 strokeWidth:(float)strokeWidth
                   hasStroke:(BOOL)hasStroke
                     hasFill:(BOOL)hasFill
              operationCount:(NSInteger)operationCount;
@end

@interface POXSearchResult ()
- (instancetype)initWithText:(NSString*)text page:(NSInteger)page bbox:(POXBbox)bbox;
@end

@interface POXRenderedImage ()
// Takes ownership of a non-null FfiRenderedImage handle; reads width/height/data
// eagerly. The handle is retained for -saveToPath: and freed on -close/-dealloc.
- (instancetype)initWithHandle:(FfiRenderedImage*)handle;
@end

@implementation POXChar
- (instancetype)initWithCharacter:(uint32_t)character
                             bbox:(POXBbox)bbox
                         fontName:(NSString*)fontName
                         fontSize:(float)fontSize {
    if ((self = [super init])) {
        _character = character;
        _bbox = bbox;
        _fontName = [fontName copy];
        _fontSize = fontSize;
    }
    return self;
}
@end

@implementation POXWord
- (instancetype)initWithText:(NSString*)text
                        bbox:(POXBbox)bbox
                    fontName:(NSString*)fontName
                    fontSize:(float)fontSize
                        bold:(BOOL)bold {
    if ((self = [super init])) {
        _text = [text copy];
        _bbox = bbox;
        _fontName = [fontName copy];
        _fontSize = fontSize;
        _bold = bold;
    }
    return self;
}
@end

@implementation POXTextLine
- (instancetype)initWithText:(NSString*)text
                        bbox:(POXBbox)bbox
                   wordCount:(NSInteger)wordCount {
    if ((self = [super init])) {
        _text = [text copy];
        _bbox = bbox;
        _wordCount = wordCount;
    }
    return self;
}
@end

@implementation POXTable {
    NSArray<NSArray<NSString*>*>* _cells;
}
- (instancetype)initWithRowCount:(NSInteger)rowCount
                        colCount:(NSInteger)colCount
                       hasHeader:(BOOL)hasHeader
                           cells:(NSArray<NSArray<NSString*>*>*)cells {
    if ((self = [super init])) {
        _rowCount = rowCount;
        _colCount = colCount;
        _hasHeader = hasHeader;
        _cells = [cells copy];
    }
    return self;
}
- (NSString*)cellTextAtRow:(NSInteger)row col:(NSInteger)col {
    if (row < 0 || row >= (NSInteger)_cells.count)
        return nil;
    NSArray<NSString*>* r = _cells[row];
    if (col < 0 || col >= (NSInteger)r.count)
        return nil;
    return r[col];
}
@end

@implementation POXFont
- (instancetype)initWithName:(NSString*)name
                        type:(NSString*)type
                    encoding:(NSString*)encoding
                    embedded:(BOOL)embedded
                      subset:(BOOL)subset {
    if ((self = [super init])) {
        _name = [name copy];
        _type = [type copy];
        _encoding = [encoding copy];
        _embedded = embedded;
        _subset = subset;
    }
    return self;
}
@end

@implementation POXImage
- (instancetype)initWithWidth:(NSInteger)width
                       height:(NSInteger)height
             bitsPerComponent:(NSInteger)bitsPerComponent
                       format:(NSString*)format
                   colorspace:(NSString*)colorspace
                         data:(NSData*)data {
    if ((self = [super init])) {
        _width = width;
        _height = height;
        _bitsPerComponent = bitsPerComponent;
        _format = [format copy];
        _colorspace = [colorspace copy];
        _data = [data copy];
    }
    return self;
}
@end

@implementation POXAnnotation
- (instancetype)initWithType:(NSString*)type
                     subtype:(NSString*)subtype
                     content:(NSString*)content
                      author:(NSString*)author
                        rect:(POXBbox)rect
                 borderWidth:(float)borderWidth {
    if ((self = [super init])) {
        _type = [type copy];
        _subtype = [subtype copy];
        _content = [content copy];
        _author = [author copy];
        _rect = rect;
        _borderWidth = borderWidth;
        _quadPoints = @[];
    }
    return self;
}
@end

@implementation POXFormField
- (instancetype)initWithName:(NSString*)name
                       value:(NSString*)value
                        type:(NSString*)type
                    readonly:(BOOL)readonly
                    required:(BOOL)required {
    if ((self = [super init])) {
        _name = [name copy];
        _value = [value copy];
        _type = [type copy];
        _readonly = readonly;
        _required = required;
    }
    return self;
}
@end

@implementation POXPath
- (instancetype)initWithBbox:(POXBbox)bbox
                 strokeWidth:(float)strokeWidth
                   hasStroke:(BOOL)hasStroke
                     hasFill:(BOOL)hasFill
              operationCount:(NSInteger)operationCount {
    if ((self = [super init])) {
        _bbox = bbox;
        _strokeWidth = strokeWidth;
        _hasStroke = hasStroke;
        _hasFill = hasFill;
        _operationCount = operationCount;
    }
    return self;
}
@end

@implementation POXSearchResult
- (instancetype)initWithText:(NSString*)text page:(NSInteger)page bbox:(POXBbox)bbox {
    if ((self = [super init])) {
        _text = [text copy];
        _page = page;
        _bbox = bbox;
    }
    return self;
}
@end

@implementation POXRenderedImage {
    FfiRenderedImage* _handle;
}
- (instancetype)initWithHandle:(FfiRenderedImage*)handle {
    if ((self = [super init])) {
        _handle = handle;
        int32_t c = 0;
        _width = pdf_get_rendered_image_width(handle, &c);
        _height = pdf_get_rendered_image_height(handle, &c);
        int32_t dataLen = 0;
        uint8_t* p = pdf_get_rendered_image_data(handle, &dataLen, &c);
        _data = p ? [NSData dataWithBytes:p
                                   length:(dataLen < 0 ? 0 : (NSUInteger)dataLen)]
                  : [NSData data];
        if (p)
            free_bytes(p);
    }
    return self;
}
- (BOOL)saveToPath:(NSString*)path error:(NSError**)error {
    if (!_handle) {
        if (error)
            *error = POXMakeError(0, @"saveRenderedImage");
        return NO;
    }
    int32_t code = 0;
    if (pdf_save_rendered_image(_handle, path.UTF8String, &code) != 0) {
        if (error)
            *error = POXMakeError(code, @"saveRenderedImage");
        return NO;
    }
    return YES;
}
- (void)close {
    if (_handle) {
        pdf_rendered_image_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_rendered_image_free(_handle);
}
@end

@implementation POXDocument {
    PdfDocument* _handle;
}

+ (instancetype)openPath:(NSString*)path error:(NSError**)error {
    int32_t code = 0;
    PdfDocument* h = pdf_document_open(path.UTF8String, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"open");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

+ (instancetype)openFromBytes:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    PdfDocument* h = pdf_document_open_from_bytes(data.bytes, data.length, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"openFromBytes");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

+ (instancetype)openWithPassword:(NSString*)path
                        password:(NSString*)password
                           error:(NSError**)error {
    int32_t code = 0;
    PdfDocument* h =
        pdf_document_open_with_password(path.UTF8String, password.UTF8String, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"openWithPassword");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

- (instancetype)initWithHandle:(PdfDocument*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (void)dealloc {
    if (_handle)
        pdf_document_free(_handle);
}

- (NSInteger)pageCountError:(NSError**)error {
    int32_t code = 0;
    int32_t n = pdf_document_get_page_count(_handle, &code);
    if (n < 0) {
        if (error)
            *error = POXMakeError(code, @"pageCount");
        return -1;
    }
    return n;
}

- (POXVersion)version {
    POXVersion v = {0, 0};
    pdf_document_get_version(_handle, &v.major, &v.minor);
    return v;
}

- (BOOL)isEncrypted {
    return pdf_document_is_encrypted(_handle);
}
- (BOOL)hasStructureTree {
    return pdf_document_has_structure_tree(_handle);
}

- (NSString*)extractText:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_extract_text(_handle, (int32_t)page, &code), code,
                         @"extractText", error);
}
- (NSString*)toPlainText:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_to_plain_text(_handle, (int32_t)page, &code),
                         code, @"toPlainText", error);
}
- (NSString*)toMarkdown:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_to_markdown(_handle, (int32_t)page, &code), code,
                         @"toMarkdown", error);
}
- (NSString*)toHtml:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_to_html(_handle, (int32_t)page, &code), code,
                         @"toHtml", error);
}
- (NSString*)toMarkdownAllWithError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_to_markdown_all(_handle, &code), code,
                         @"toMarkdownAll", error);
}
- (NSString*)toHtmlAllWithError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_to_html_all(_handle, &code), code, @"toHtmlAll",
                         error);
}
- (NSString*)toPlainTextAllWithError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_to_plain_text_all(_handle, &code), code,
                         @"toPlainTextAll", error);
}
- (BOOL)authenticate:(NSString*)password error:(NSError**)error {
    int32_t code = 0;
    bool ok = pdf_document_authenticate(_handle, password.UTF8String, &code);
    if (!ok && code != 0) {
        if (error)
            *error = POXMakeError(code, @"authenticate");
    }
    return ok ? YES : NO;
}
- (POXPage*)pageAtIndex:(NSInteger)index {
    return [[POXPage alloc] initWithDocument:self index:index];
}
- (NSString*)extractStructuredJson:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(
        pdf_document_extract_structured_to_json(_handle, (int32_t)page, &code), code,
        @"extractStructuredJson", error);
}

- (NSArray<POXChar*>*)extractChars:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    FfiCharList* list = pdf_document_extract_chars(_handle, (int32_t)page, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"extractChars");
        return nil;
    }
    int32_t n = pdf_oxide_char_count(list);
    NSMutableArray<POXChar*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        uint32_t ch = pdf_oxide_char_get_char(list, i, &c);
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_char_get_bbox(list, i, &x, &y, &w, &h, &c);
        NSString* fontName = POXTakeString(pdf_oxide_char_get_font_name(list, i, &c), c,
                                           @"charFontName", NULL);
        float fontSize = pdf_oxide_char_get_font_size(list, i, &c);
        POXBbox bbox = {x, y, w, h};
        [out addObject:[[POXChar alloc]
                           initWithCharacter:ch
                                        bbox:bbox
                                    fontName:(fontName ?: @"")fontSize:fontSize]];
    }
    pdf_oxide_char_list_free(list);
    return out;
}

- (NSArray<POXWord*>*)extractWords:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    FfiWordList* list = pdf_document_extract_words(_handle, (int32_t)page, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"extractWords");
        return nil;
    }
    int32_t n = pdf_oxide_word_count(list);
    NSMutableArray<POXWord*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* text =
            POXTakeString(pdf_oxide_word_get_text(list, i, &c), c, @"wordText", NULL);
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_word_get_bbox(list, i, &x, &y, &w, &h, &c);
        NSString* fontName = POXTakeString(pdf_oxide_word_get_font_name(list, i, &c), c,
                                           @"wordFontName", NULL);
        float fontSize = pdf_oxide_word_get_font_size(list, i, &c);
        bool bold = pdf_oxide_word_is_bold(list, i, &c);
        POXBbox bbox = {x, y, w, h};
        [out addObject:[[POXWord alloc] initWithText:(text ?: @"")
                                                bbox:bbox
                                            fontName:(fontName ?: @"")fontSize:fontSize
                                                bold:(bold ? YES : NO)]];
    }
    pdf_oxide_word_list_free(list);
    return out;
}

- (NSArray<POXTextLine*>*)extractTextLines:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    FfiTextLineList* list =
        pdf_document_extract_text_lines(_handle, (int32_t)page, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"extractTextLines");
        return nil;
    }
    int32_t n = pdf_oxide_line_count(list);
    NSMutableArray<POXTextLine*>* out =
        [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* text =
            POXTakeString(pdf_oxide_line_get_text(list, i, &c), c, @"lineText", NULL);
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_line_get_bbox(list, i, &x, &y, &w, &h, &c);
        int32_t wordCount = pdf_oxide_line_get_word_count(list, i, &c);
        POXBbox bbox = {x, y, w, h};
        [out addObject:[[POXTextLine alloc] initWithText:(text ?: @"")
                                                    bbox:bbox
                                               wordCount:wordCount]];
    }
    pdf_oxide_line_list_free(list);
    return out;
}

- (NSArray<POXTable*>*)extractTables:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    FfiTableList* list = pdf_document_extract_tables(_handle, (int32_t)page, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"extractTables");
        return nil;
    }
    int32_t n = pdf_oxide_table_count(list);
    NSMutableArray<POXTable*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        int32_t rowCount = pdf_oxide_table_get_row_count(list, i, &c);
        int32_t colCount = pdf_oxide_table_get_col_count(list, i, &c);
        bool hasHeader = pdf_oxide_table_has_header(list, i, &c);
        NSMutableArray<NSArray<NSString*>*>* cells =
            [NSMutableArray arrayWithCapacity:(rowCount < 0 ? 0 : rowCount)];
        for (int32_t r = 0; r < rowCount; ++r) {
            NSMutableArray<NSString*>* row =
                [NSMutableArray arrayWithCapacity:(colCount < 0 ? 0 : colCount)];
            for (int32_t col = 0; col < colCount; ++col) {
                NSString* cell =
                    POXTakeString(pdf_oxide_table_get_cell_text(list, i, r, col, &c), c,
                                  @"tableCell", NULL);
                [row addObject:(cell ?: @"")];
            }
            [cells addObject:row];
        }
        [out addObject:[[POXTable alloc]
                           initWithRowCount:rowCount
                                   colCount:colCount
                                  hasHeader:(hasHeader ? YES : NO)cells:cells]];
    }
    pdf_oxide_table_list_free(list);
    return out;
}

- (NSArray<POXFont*>*)embeddedFonts:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    FfiFontList* list = pdf_document_get_embedded_fonts(_handle, (int32_t)page, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"embeddedFonts");
        return nil;
    }
    int32_t n = pdf_oxide_font_count(list);
    NSMutableArray<POXFont*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* name =
            POXTakeString(pdf_oxide_font_get_name(list, i, &c), c, @"fontName", NULL);
        NSString* type =
            POXTakeString(pdf_oxide_font_get_type(list, i, &c), c, @"fontType", NULL);
        NSString* encoding = POXTakeString(pdf_oxide_font_get_encoding(list, i, &c), c,
                                           @"fontEncoding", NULL);
        bool embedded = pdf_oxide_font_is_embedded(list, i, &c) != 0;
        bool subset = pdf_oxide_font_is_subset(list, i, &c) != 0;
        float size = pdf_oxide_font_get_size(list, i, &c);
        POXFont* f = [[POXFont alloc]
            initWithName:(name ?: @"")
                    type:(type ?: @"")encoding:(encoding ?: @"")embedded
                        :(embedded ? YES : NO)subset:(subset ? YES : NO)];
        f.size = size;
        [out addObject:f];
    }
    pdf_oxide_font_list_free(list);
    return out;
}

- (NSString*)embeddedFontsJson:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    FfiFontList* list = pdf_document_get_embedded_fonts(_handle, (int32_t)page, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"embeddedFontsJson");
        return nil;
    }
    NSString* json = POXTakeString(pdf_oxide_fonts_to_json(list, &code), code,
                                   @"fontsToJson", error);
    pdf_oxide_font_list_free(list);
    return json;
}

- (NSArray<POXImage*>*)embeddedImages:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    FfiImageList* list =
        pdf_document_get_embedded_images(_handle, (int32_t)page, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"embeddedImages");
        return nil;
    }
    int32_t n = pdf_oxide_image_count(list);
    NSMutableArray<POXImage*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        int32_t width = pdf_oxide_image_get_width(list, i, &c);
        int32_t height = pdf_oxide_image_get_height(list, i, &c);
        int32_t bpc = pdf_oxide_image_get_bits_per_component(list, i, &c);
        NSString* format = POXTakeString(pdf_oxide_image_get_format(list, i, &c), c,
                                         @"imageFormat", NULL);
        NSString* colorspace = POXTakeString(
            pdf_oxide_image_get_colorspace(list, i, &c), c, @"imageColorspace", NULL);
        int32_t dataLen = 0;
        uint8_t* p = pdf_oxide_image_get_data(list, i, &dataLen, &c);
        NSData* data =
            p ? [NSData dataWithBytes:p length:(dataLen < 0 ? 0 : (NSUInteger)dataLen)]
              : [NSData data];
        if (p)
            free_bytes(p);
        [out addObject:[[POXImage alloc] initWithWidth:width
                                                height:height
                                      bitsPerComponent:bpc
                                                format:(format ?: @"")colorspace
                                                      :(colorspace ?: @"")data:data]];
    }
    pdf_oxide_image_list_free(list);
    return out;
}

- (NSArray<POXAnnotation*>*)pageAnnotations:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    FfiAnnotationList* list =
        pdf_document_get_page_annotations(_handle, (int32_t)page, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"pageAnnotations");
        return nil;
    }
    int32_t n = pdf_oxide_annotation_count(list);
    NSMutableArray<POXAnnotation*>* out =
        [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* type = POXTakeString(pdf_oxide_annotation_get_type(list, i, &c), c,
                                       @"annotationType", NULL);
        NSString* subtype = POXTakeString(pdf_oxide_annotation_get_subtype(list, i, &c),
                                          c, @"annotationSubtype", NULL);
        NSString* content = POXTakeString(pdf_oxide_annotation_get_content(list, i, &c),
                                          c, @"annotationContent", NULL);
        NSString* author = POXTakeString(pdf_oxide_annotation_get_author(list, i, &c),
                                         c, @"annotationAuthor", NULL);
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_annotation_get_rect(list, i, &x, &y, &w, &h, &c);
        float borderWidth = pdf_oxide_annotation_get_border_width(list, i, &c);
        POXBbox rect = {x, y, w, h};
        POXAnnotation* a = [[POXAnnotation alloc]
            initWithType:(type ?: @"")
                 subtype:(subtype ?: @"")content:(content ?: @"")author
                        :(author ?: @"")rect:rect
             borderWidth:borderWidth];
        a.color = pdf_oxide_annotation_get_color(list, i, &c);
        a.creationDate = pdf_oxide_annotation_get_creation_date(list, i, &c);
        a.modificationDate = pdf_oxide_annotation_get_modification_date(list, i, &c);
        a.hidden = pdf_oxide_annotation_is_hidden(list, i, &c) ? YES : NO;
        a.markedDeleted =
            pdf_oxide_annotation_is_marked_deleted(list, i, &c) ? YES : NO;
        a.printable = pdf_oxide_annotation_is_printable(list, i, &c) ? YES : NO;
        a.readOnly = pdf_oxide_annotation_is_read_only(list, i, &c) ? YES : NO;
        a.linkUri = POXTakeString(pdf_oxide_link_annotation_get_uri(list, i, &c), c,
                                  @"linkUri", NULL);
        a.iconName = POXTakeString(pdf_oxide_text_annotation_get_icon_name(list, i, &c),
                                   c, @"iconName", NULL);
        int32_t quadCount =
            pdf_oxide_highlight_annotation_get_quad_points_count(list, i, &c);
        if (quadCount > 0) {
            NSMutableArray<NSArray<NSNumber*>*>* quads =
                [NSMutableArray arrayWithCapacity:(NSUInteger)quadCount];
            for (int32_t q = 0; q < quadCount; ++q) {
                float x1 = 0, y1 = 0, x2 = 0, y2 = 0, x3 = 0, y3 = 0, x4 = 0, y4 = 0;
                pdf_oxide_highlight_annotation_get_quad_point(
                    list, i, q, &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4, &c);
                [quads addObject:@[
                    @(x1), @(y1), @(x2), @(y2), @(x3), @(y3), @(x4), @(y4)
                ]];
            }
            a.quadPoints = quads;
        }
        [out addObject:a];
    }
    pdf_oxide_annotation_list_free(list);
    return out;
}

- (NSString*)annotationsJson:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    FfiAnnotationList* list =
        pdf_document_get_page_annotations(_handle, (int32_t)page, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"annotationsJson");
        return nil;
    }
    NSString* json = POXTakeString(pdf_oxide_annotations_to_json(list, &code), code,
                                   @"annotationsToJson", error);
    pdf_oxide_annotation_list_free(list);
    return json;
}

- (NSArray<POXPath*>*)extractPaths:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    FfiPathList* list = pdf_document_extract_paths(_handle, (int32_t)page, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"extractPaths");
        return nil;
    }
    int32_t n = pdf_oxide_path_count(list);
    NSMutableArray<POXPath*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_path_get_bbox(list, i, &x, &y, &w, &h, &c);
        float strokeWidth = pdf_oxide_path_get_stroke_width(list, i, &c);
        bool hasStroke = pdf_oxide_path_has_stroke(list, i, &c);
        bool hasFill = pdf_oxide_path_has_fill(list, i, &c);
        int32_t operationCount = pdf_oxide_path_get_operation_count(list, i, &c);
        POXBbox bbox = {x, y, w, h};
        [out addObject:[[POXPath alloc] initWithBbox:bbox
                                         strokeWidth:strokeWidth
                                           hasStroke:(hasStroke ? YES : NO)hasFill
                                                    :(hasFill ? YES : NO)operationCount
                                                    :operationCount]];
    }
    pdf_oxide_path_list_free(list);
    return out;
}

// Marshal a FfiSearchResults handle into an array of POXSearchResult, then free it.
static NSArray<POXSearchResult*>* POXTakeSearchResults(FfiSearchResults* list) {
    int32_t n = pdf_oxide_search_result_count(list);
    NSMutableArray<POXSearchResult*>* out =
        [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* text = POXTakeString(pdf_oxide_search_result_get_text(list, i, &c), c,
                                       @"searchResultText", NULL);
        int32_t resultPage = pdf_oxide_search_result_get_page(list, i, &c);
        float x = 0, y = 0, w = 0, h = 0;
        pdf_oxide_search_result_get_bbox(list, i, &x, &y, &w, &h, &c);
        POXBbox bbox = {x, y, w, h};
        [out addObject:[[POXSearchResult alloc] initWithText:(text ?: @"")
                                                        page:resultPage
                                                        bbox:bbox]];
    }
    pdf_oxide_search_result_free(list);
    return out;
}

- (NSArray<POXSearchResult*>*)search:(NSInteger)page
                                term:(NSString*)term
                       caseSensitive:(BOOL)caseSensitive
                               error:(NSError**)error {
    int32_t code = 0;
    FfiSearchResults* list = pdf_document_search_page(
        _handle, (int32_t)page, term.UTF8String, caseSensitive ? true : false, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"search");
        return nil;
    }
    return POXTakeSearchResults(list);
}

- (NSArray<POXSearchResult*>*)searchAll:(NSString*)term
                          caseSensitive:(BOOL)caseSensitive
                                  error:(NSError**)error {
    int32_t code = 0;
    FfiSearchResults* list = pdf_document_search_all(
        _handle, term.UTF8String, caseSensitive ? true : false, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"searchAll");
        return nil;
    }
    return POXTakeSearchResults(list);
}

- (NSString*)searchJson:(NSInteger)page
                   term:(NSString*)term
          caseSensitive:(BOOL)caseSensitive
                  error:(NSError**)error {
    int32_t code = 0;
    FfiSearchResults* list = pdf_document_search_page(
        _handle, (int32_t)page, term.UTF8String, caseSensitive ? true : false, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"searchJson");
        return nil;
    }
    NSString* json = POXTakeString(pdf_oxide_search_results_to_json(list, &code), code,
                                   @"searchResultsToJson", error);
    pdf_oxide_search_result_free(list);
    return json;
}

- (POXRenderedImage*)renderPage:(NSInteger)pageIndex
                         format:(int32_t)format
                          error:(NSError**)error {
    int32_t code = 0;
    FfiRenderedImage* img = pdf_render_page(_handle, (int32_t)pageIndex, format, &code);
    if (!img) {
        if (error)
            *error = POXMakeError(code, @"renderPage");
        return nil;
    }
    return [[POXRenderedImage alloc] initWithHandle:img];
}

- (POXRenderedImage*)renderPageZoom:(NSInteger)pageIndex
                               zoom:(float)zoom
                             format:(int32_t)format
                              error:(NSError**)error {
    int32_t code = 0;
    FfiRenderedImage* img =
        pdf_render_page_zoom(_handle, (int32_t)pageIndex, zoom, format, &code);
    if (!img) {
        if (error)
            *error = POXMakeError(code, @"renderPageZoom");
        return nil;
    }
    return [[POXRenderedImage alloc] initWithHandle:img];
}

- (POXRenderedImage*)renderPageThumbnail:(NSInteger)pageIndex
                                    size:(int32_t)size
                                  format:(int32_t)format
                                   error:(NSError**)error {
    int32_t code = 0;
    FfiRenderedImage* img =
        pdf_render_page_thumbnail(_handle, (int32_t)pageIndex, size, format, &code);
    if (!img) {
        if (error)
            *error = POXMakeError(code, @"renderPageThumbnail");
        return nil;
    }
    return [[POXRenderedImage alloc] initWithHandle:img];
}

- (POXPdfAResults*)validatePdfA:(int32_t)level error:(NSError**)error {
    int32_t code = 0;
    FfiPdfAResults* r = pdf_validate_pdf_a_level(_handle, level, &code);
    if (!r) {
        if (error)
            *error = POXMakeError(code, @"validatePdfA");
        return nil;
    }
    return [[POXPdfAResults alloc] initWithHandle:r];
}

- (POXUaResults*)validatePdfUa:(int32_t)level error:(NSError**)error {
    int32_t code = 0;
    FfiUaResults* r = pdf_validate_pdf_ua(_handle, level, &code);
    if (!r) {
        if (error)
            *error = POXMakeError(code, @"validatePdfUa");
        return nil;
    }
    return [[POXUaResults alloc] initWithHandle:r];
}

- (POXPdfXResults*)validatePdfX:(int32_t)level error:(NSError**)error {
    int32_t code = 0;
    FfiPdfXResults* r = pdf_validate_pdf_x_level(_handle, level, &code);
    if (!r) {
        if (error)
            *error = POXMakeError(code, @"validatePdfX");
        return nil;
    }
    return [[POXPdfXResults alloc] initWithHandle:r];
}

// ── Phase-7: render variants / page getters / OCR ────────────────────────────

- (POXRenderedImage*)renderPageWithOptions:(NSInteger)pageIndex
                                       dpi:(int32_t)dpi
                                    format:(int32_t)format
                                       bgR:(float)bgR
                                       bgG:(float)bgG
                                       bgB:(float)bgB
                                       bgA:(float)bgA
                     transparentBackground:(int32_t)transparentBackground
                         renderAnnotations:(int32_t)renderAnnotations
                               jpegQuality:(int32_t)jpegQuality
                                     error:(NSError**)error {
    int32_t code = 0;
    FfiRenderedImage* img = pdf_render_page_with_options(
        _handle, (int32_t)pageIndex, dpi, format, bgR, bgG, bgB, bgA,
        transparentBackground, renderAnnotations, jpegQuality, &code);
    if (!img) {
        if (error)
            *error = POXMakeError(code, @"renderPageWithOptions");
        return nil;
    }
    return [[POXRenderedImage alloc] initWithHandle:img];
}

- (POXRenderedImage*)renderPageWithOptionsEx:(NSInteger)pageIndex
                                         dpi:(int32_t)dpi
                                      format:(int32_t)format
                                         bgR:(float)bgR
                                         bgG:(float)bgG
                                         bgB:(float)bgB
                                         bgA:(float)bgA
                       transparentBackground:(int32_t)transparentBackground
                           renderAnnotations:(int32_t)renderAnnotations
                                 jpegQuality:(int32_t)jpegQuality
                              excludedLayers:(NSArray<NSString*>*)excludedLayers
                                       error:(NSError**)error {
    NSUInteger count = excludedLayers.count;
    const char** layers = count ? (const char**)malloc(sizeof(char*) * count) : NULL;
    for (NSUInteger i = 0; i < count; ++i)
        layers[i] = excludedLayers[i].UTF8String;
    int32_t code = 0;
    FfiRenderedImage* img = pdf_render_page_with_options_ex(
        _handle, (int32_t)pageIndex, dpi, format, bgR, bgG, bgB, bgA,
        transparentBackground, renderAnnotations, jpegQuality, layers, (uintptr_t)count,
        &code);
    if (layers)
        free(layers);
    if (!img) {
        if (error)
            *error = POXMakeError(code, @"renderPageWithOptionsEx");
        return nil;
    }
    return [[POXRenderedImage alloc] initWithHandle:img];
}

- (POXRenderedImage*)renderPageRegion:(NSInteger)pageIndex
                                cropX:(float)cropX
                                cropY:(float)cropY
                            cropWidth:(float)cropWidth
                           cropHeight:(float)cropHeight
                               format:(int32_t)format
                                error:(NSError**)error {
    int32_t code = 0;
    FfiRenderedImage* img =
        pdf_render_page_region(_handle, (int32_t)pageIndex, cropX, cropY, cropWidth,
                               cropHeight, format, &code);
    if (!img) {
        if (error)
            *error = POXMakeError(code, @"renderPageRegion");
        return nil;
    }
    return [[POXRenderedImage alloc] initWithHandle:img];
}

- (POXRenderedImage*)renderPageFit:(NSInteger)pageIndex
                                 w:(int32_t)w
                                 h:(int32_t)h
                            format:(int32_t)format
                             error:(NSError**)error {
    int32_t code = 0;
    FfiRenderedImage* img =
        pdf_render_page_fit(_handle, (int32_t)pageIndex, w, h, format, &code);
    if (!img) {
        if (error)
            *error = POXMakeError(code, @"renderPageFit");
        return nil;
    }
    return [[POXRenderedImage alloc] initWithHandle:img];
}

- (POXRenderedImage*)renderPageRaw:(NSInteger)pageIndex
                               dpi:(int32_t)dpi
                          outWidth:(int32_t*)outWidth
                         outHeight:(int32_t*)outHeight
                             error:(NSError**)error {
    int32_t code = 0;
    int32_t w = 0, h = 0;
    FfiRenderedImage* img =
        pdf_render_page_raw(_handle, (int32_t)pageIndex, dpi, &w, &h, &code);
    if (!img) {
        if (error)
            *error = POXMakeError(code, @"renderPageRaw");
        return nil;
    }
    if (outWidth)
        *outWidth = w;
    if (outHeight)
        *outHeight = h;
    return [[POXRenderedImage alloc] initWithHandle:img];
}

- (int32_t)estimateRenderTime:(NSInteger)pageIndex error:(NSError**)error {
    int32_t code = 0;
    int32_t t = pdf_estimate_render_time(_handle, (int32_t)pageIndex, &code);
    if (t < 0 && error)
        *error = POXMakeError(code, @"estimateRenderTime");
    return t;
}

- (float)pageWidth:(NSInteger)pageIndex error:(NSError**)error {
    int32_t code = 0;
    float v = pdf_page_get_width(_handle, (int32_t)pageIndex, &code);
    if (code != 0 && error)
        *error = POXMakeError(code, @"pageWidth");
    return v;
}

- (float)pageHeight:(NSInteger)pageIndex error:(NSError**)error {
    int32_t code = 0;
    float v = pdf_page_get_height(_handle, (int32_t)pageIndex, &code);
    if (code != 0 && error)
        *error = POXMakeError(code, @"pageHeight");
    return v;
}

- (int32_t)pageRotation:(NSInteger)pageIndex error:(NSError**)error {
    int32_t code = 0;
    int32_t r = pdf_page_get_rotation(_handle, (int32_t)pageIndex, &code);
    if (r < 0 && error)
        *error = POXMakeError(code, @"pageRotation");
    return r;
}

- (POXElementList*)pageElements:(NSInteger)pageIndex error:(NSError**)error {
    int32_t code = 0;
    FfiElementList* list = pdf_page_get_elements(_handle, (int32_t)pageIndex, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"pageElements");
        return nil;
    }
    return [[POXElementList alloc] initWithHandle:list];
}

- (BOOL)pageNeedsOcr:(NSInteger)pageIndex error:(NSError**)error {
    int32_t code = 0;
    bool needs = pdf_ocr_page_needs_ocr(_handle, (int32_t)pageIndex, &code);
    if (code != 0 && error)
        *error = POXMakeError(code, @"pageNeedsOcr");
    return needs ? YES : NO;
}

- (NSString*)ocrExtractText:(NSInteger)pageIndex
                     engine:(POXOcrEngine*)engine
                      error:(NSError**)error {
    int32_t code = 0;
    const void* eng = engine ? [engine POX_engineHandle] : NULL;
    return POXTakeString(pdf_ocr_extract_text(_handle, (int32_t)pageIndex, eng, &code),
                         code, @"ocrExtractText", error);
}

// ── Office open/export ───────────────────────────────────────────────────────

+ (instancetype)openFromDocxBytes:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    PdfDocument* h =
        pdf_document_open_from_docx_bytes(data.bytes, (uintptr_t)data.length, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"openFromDocxBytes");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}
+ (instancetype)openFromPptxBytes:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    PdfDocument* h =
        pdf_document_open_from_pptx_bytes(data.bytes, (uintptr_t)data.length, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"openFromPptxBytes");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}
+ (instancetype)openFromXlsxBytes:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    PdfDocument* h =
        pdf_document_open_from_xlsx_bytes(data.bytes, (uintptr_t)data.length, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"openFromXlsxBytes");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

- (NSData*)toDocxWithError:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = pdf_document_to_docx(_handle, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"toDocx", error);
}
- (NSData*)toPptxWithError:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = pdf_document_to_pptx(_handle, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"toPptx", error);
}
- (NSData*)toXlsxWithError:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = pdf_document_to_xlsx(_handle, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"toXlsx", error);
}

// ── In-rect extractors ───────────────────────────────────────────────────────

- (NSString*)extractTextInRect:(NSInteger)page
                             x:(float)x
                             y:(float)y
                         width:(float)width
                        height:(float)height
                         error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_extract_text_in_rect(_handle, (int32_t)page, x, y,
                                                           width, height, &code),
                         code, @"extractTextInRect", error);
}

- (NSArray<POXWord*>*)extractWordsInRect:(NSInteger)page
                                       x:(float)x
                                       y:(float)y
                                   width:(float)width
                                  height:(float)height
                                   error:(NSError**)error {
    int32_t code = 0;
    FfiWordList* list = pdf_document_extract_words_in_rect(_handle, (int32_t)page, x, y,
                                                           width, height, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"extractWordsInRect");
        return nil;
    }
    int32_t n = pdf_oxide_word_count(list);
    NSMutableArray<POXWord*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* text =
            POXTakeString(pdf_oxide_word_get_text(list, i, &c), c, @"wordText", NULL);
        float bx = 0, by = 0, bw = 0, bh = 0;
        pdf_oxide_word_get_bbox(list, i, &bx, &by, &bw, &bh, &c);
        NSString* fontName = POXTakeString(pdf_oxide_word_get_font_name(list, i, &c), c,
                                           @"wordFontName", NULL);
        float fontSize = pdf_oxide_word_get_font_size(list, i, &c);
        bool bold = pdf_oxide_word_is_bold(list, i, &c);
        POXBbox bbox = {bx, by, bw, bh};
        [out addObject:[[POXWord alloc] initWithText:(text ?: @"")
                                                bbox:bbox
                                            fontName:(fontName ?: @"")fontSize:fontSize
                                                bold:(bold ? YES : NO)]];
    }
    pdf_oxide_word_list_free(list);
    return out;
}

- (NSArray<POXTextLine*>*)extractLinesInRect:(NSInteger)page
                                           x:(float)x
                                           y:(float)y
                                       width:(float)width
                                      height:(float)height
                                       error:(NSError**)error {
    int32_t code = 0;
    FfiTextLineList* list = pdf_document_extract_lines_in_rect(
        _handle, (int32_t)page, x, y, width, height, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"extractLinesInRect");
        return nil;
    }
    int32_t n = pdf_oxide_line_count(list);
    NSMutableArray<POXTextLine*>* out =
        [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* text =
            POXTakeString(pdf_oxide_line_get_text(list, i, &c), c, @"lineText", NULL);
        float bx = 0, by = 0, bw = 0, bh = 0;
        pdf_oxide_line_get_bbox(list, i, &bx, &by, &bw, &bh, &c);
        int32_t wordCount = pdf_oxide_line_get_word_count(list, i, &c);
        POXBbox bbox = {bx, by, bw, bh};
        [out addObject:[[POXTextLine alloc] initWithText:(text ?: @"")
                                                    bbox:bbox
                                               wordCount:wordCount]];
    }
    pdf_oxide_line_list_free(list);
    return out;
}

- (NSArray<POXTable*>*)extractTablesInRect:(NSInteger)page
                                         x:(float)x
                                         y:(float)y
                                     width:(float)width
                                    height:(float)height
                                     error:(NSError**)error {
    int32_t code = 0;
    FfiTableList* list = pdf_document_extract_tables_in_rect(_handle, (int32_t)page, x,
                                                             y, width, height, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"extractTablesInRect");
        return nil;
    }
    int32_t n = pdf_oxide_table_count(list);
    NSMutableArray<POXTable*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        int32_t rowCount = pdf_oxide_table_get_row_count(list, i, &c);
        int32_t colCount = pdf_oxide_table_get_col_count(list, i, &c);
        bool hasHeader = pdf_oxide_table_has_header(list, i, &c);
        NSMutableArray<NSArray<NSString*>*>* cells =
            [NSMutableArray arrayWithCapacity:(rowCount < 0 ? 0 : rowCount)];
        for (int32_t r = 0; r < rowCount; ++r) {
            NSMutableArray<NSString*>* row =
                [NSMutableArray arrayWithCapacity:(colCount < 0 ? 0 : colCount)];
            for (int32_t col = 0; col < colCount; ++col) {
                NSString* cell =
                    POXTakeString(pdf_oxide_table_get_cell_text(list, i, r, col, &c), c,
                                  @"tableCell", NULL);
                [row addObject:(cell ?: @"")];
            }
            [cells addObject:row];
        }
        [out addObject:[[POXTable alloc]
                           initWithRowCount:rowCount
                                   colCount:colCount
                                  hasHeader:(hasHeader ? YES : NO)cells:cells]];
    }
    pdf_oxide_table_list_free(list);
    return out;
}

- (NSArray<POXImage*>*)extractImagesInRect:(NSInteger)page
                                         x:(float)x
                                         y:(float)y
                                     width:(float)width
                                    height:(float)height
                                     error:(NSError**)error {
    int32_t code = 0;
    FfiImageList* list = pdf_document_extract_images_in_rect(_handle, (int32_t)page, x,
                                                             y, width, height, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"extractImagesInRect");
        return nil;
    }
    int32_t n = pdf_oxide_image_count(list);
    NSMutableArray<POXImage*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        int32_t iw = pdf_oxide_image_get_width(list, i, &c);
        int32_t ih = pdf_oxide_image_get_height(list, i, &c);
        int32_t bpc = pdf_oxide_image_get_bits_per_component(list, i, &c);
        NSString* format = POXTakeString(pdf_oxide_image_get_format(list, i, &c), c,
                                         @"imageFormat", NULL);
        NSString* colorspace = POXTakeString(
            pdf_oxide_image_get_colorspace(list, i, &c), c, @"imageColorspace", NULL);
        int32_t dataLen = 0;
        uint8_t* p = pdf_oxide_image_get_data(list, i, &dataLen, &c);
        NSData* data =
            p ? [NSData dataWithBytes:p length:(dataLen < 0 ? 0 : (NSUInteger)dataLen)]
              : [NSData data];
        if (p)
            free_bytes(p);
        [out addObject:[[POXImage alloc] initWithWidth:iw
                                                height:ih
                                      bitsPerComponent:bpc
                                                format:(format ?: @"")colorspace
                                                      :(colorspace ?: @"")data:data]];
    }
    pdf_oxide_image_list_free(list);
    return out;
}

// ── Auto extraction / classification ─────────────────────────────────────────

- (NSString*)extractTextAuto:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_extract_text_auto(_handle, (int32_t)page, &code),
                         code, @"extractTextAuto", error);
}
- (NSString*)extractAllTextWithError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_extract_all_text(_handle, &code), code,
                         @"extractAllText", error);
}
- (NSString*)extractPageAuto:(NSInteger)page
                 optionsJson:(NSString*)optionsJson
                       error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(
        pdf_document_extract_page_auto(
            _handle, (int32_t)page, optionsJson ? optionsJson.UTF8String : NULL, &code),
        code, @"extractPageAuto", error);
}
- (NSString*)classifyPage:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_classify_page(_handle, (int32_t)page, &code),
                         code, @"classifyPage", error);
}
- (NSString*)classifyDocumentWithError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_classify_document(_handle, &code), code,
                         @"classifyDocument", error);
}

// ── Header / footer / artifact removal ───────────────────────────────────────

- (int32_t)eraseHeader:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    int32_t n = pdf_document_erase_header(_handle, (int32_t)page, &code);
    if (n < 0 && error)
        *error = POXMakeError(code, @"eraseHeader");
    return n;
}
- (int32_t)eraseFooter:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    int32_t n = pdf_document_erase_footer(_handle, (int32_t)page, &code);
    if (n < 0 && error)
        *error = POXMakeError(code, @"eraseFooter");
    return n;
}
- (int32_t)eraseArtifacts:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    int32_t n = pdf_document_erase_artifacts(_handle, (int32_t)page, &code);
    if (n < 0 && error)
        *error = POXMakeError(code, @"eraseArtifacts");
    return n;
}
- (int32_t)removeHeaders:(float)threshold error:(NSError**)error {
    int32_t code = 0;
    int32_t n = pdf_document_remove_headers(_handle, threshold, &code);
    if (n < 0 && error)
        *error = POXMakeError(code, @"removeHeaders");
    return n;
}
- (int32_t)removeFooters:(float)threshold error:(NSError**)error {
    int32_t code = 0;
    int32_t n = pdf_document_remove_footers(_handle, threshold, &code);
    if (n < 0 && error)
        *error = POXMakeError(code, @"removeFooters");
    return n;
}
- (int32_t)removeArtifacts:(float)threshold error:(NSError**)error {
    int32_t code = 0;
    int32_t n = pdf_document_remove_artifacts(_handle, threshold, &code);
    if (n < 0 && error)
        *error = POXMakeError(code, @"removeArtifacts");
    return n;
}

// ── AcroForm fields & form data ──────────────────────────────────────────────

- (NSArray<POXFormField*>*)formFieldsWithError:(NSError**)error {
    int32_t code = 0;
    FfiFormFieldList* list = pdf_document_get_form_fields(_handle, &code);
    if (!list) {
        if (error)
            *error = POXMakeError(code, @"formFields");
        return nil;
    }
    int32_t n = pdf_oxide_form_field_count(list);
    NSMutableArray<POXFormField*>* out =
        [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* name = POXTakeString(pdf_oxide_form_field_get_name(list, i, &c), c,
                                       @"formFieldName", NULL);
        NSString* value = POXTakeString(pdf_oxide_form_field_get_value(list, i, &c), c,
                                        @"formFieldValue", NULL);
        NSString* ftype = POXTakeString(pdf_oxide_form_field_get_type(list, i, &c), c,
                                        @"formFieldType", NULL);
        bool ro = pdf_oxide_form_field_is_readonly(list, i, &c);
        bool req = pdf_oxide_form_field_is_required(list, i, &c);
        [out addObject:[[POXFormField alloc]
                           initWithName:(name ?: @"")
                                  value:(value ?: @"")type:(ftype ?: @"")readonly
                                       :(ro ? YES : NO)required:(req ? YES : NO)]];
    }
    pdf_oxide_form_field_list_free(list);
    return out;
}

- (NSData*)exportFormDataToBytes:(int32_t)formatType error:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p =
        pdf_document_export_form_data_to_bytes(_handle, formatType, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"exportFormDataToBytes", error);
}

- (BOOL)importFormDataFromPath:(NSString*)dataPath error:(NSError**)error {
    int32_t code = 0;
    int32_t rc = pdf_document_import_form_data(_handle, dataPath.UTF8String, &code);
    if (rc != 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, @"importFormData");
        return NO;
    }
    return YES;
}

- (BOOL)importFormFromFile:(NSString*)filename error:(NSError**)error {
    int32_t code = 0;
    bool ok = pdf_form_import_from_file(_handle, filename.UTF8String, &code);
    if (!ok || code != 0) {
        if (error)
            *error = POXMakeError(code, @"importFormFromFile");
        return NO;
    }
    return YES;
}

// ── Document structure / metadata ────────────────────────────────────────────

- (NSString*)outlineWithError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_get_outline(_handle, &code), code, @"outline",
                         error);
}
- (NSString*)pageLabelsWithError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_get_page_labels(_handle, &code), code,
                         @"pageLabels", error);
}
- (NSString*)xmpMetadataWithError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_document_get_xmp_metadata(_handle, &code), code,
                         @"xmpMetadata", error);
}
- (NSData*)sourceBytesWithError:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = pdf_document_get_source_bytes(_handle, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"sourceBytes", error);
}
- (BOOL)hasXfa {
    return pdf_document_has_xfa(_handle) ? YES : NO;
}
- (NSString*)planSplitByBookmarks:(NSString*)optionsJson error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(
        pdf_document_plan_split_by_bookmarks(
            _handle, optionsJson ? optionsJson.UTF8String : NULL, &code),
        code, @"planSplitByBookmarks", error);
}

// ── PDF/A conversion ─────────────────────────────────────────────────────────

- (BOOL)convertToPdfA:(int32_t)level error:(NSError**)error {
    int32_t code = 0;
    bool ok = pdf_convert_to_pdf_a(_handle, level, &code);
    if (!ok || code != 0) {
        if (error)
            *error = POXMakeError(code, @"convertToPdfA");
        return NO;
    }
    return YES;
}

// ── Signatures (document-level) ──────────────────────────────────────────────

- (BOOL)sign:(POXCertificate*)certificate
      reason:(NSString*)reason
    location:(NSString*)location
       error:(NSError**)error {
    int32_t code = 0;
    int32_t rc = pdf_document_sign(_handle, [certificate POX_handle],
                                   reason ? reason.UTF8String : NULL,
                                   location ? location.UTF8String : NULL, &code);
    if (rc != 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, @"sign");
        return NO;
    }
    return YES;
}
- (int32_t)signatureCountWithError:(NSError**)error {
    int32_t code = 0;
    int32_t n = pdf_document_get_signature_count(_handle, &code);
    if (n < 0 && error)
        *error = POXMakeError(code, @"signatureCount");
    return n;
}
- (POXSignatureInfo*)signatureAtIndex:(int32_t)index error:(NSError**)error {
    int32_t code = 0;
    void* h = pdf_document_get_signature(_handle, index, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"signature");
        return nil;
    }
    return [[POXSignatureInfo alloc] initWithHandle:(FfiSignatureInfo*)h];
}
- (int32_t)verifyAllSignaturesWithError:(NSError**)error {
    int32_t code = 0;
    int32_t r = pdf_document_verify_all_signatures(_handle, &code);
    if (r < 0 && error)
        *error = POXMakeError(code, @"verifyAllSignatures");
    return r;
}
- (int32_t)hasTimestampWithError:(NSError**)error {
    int32_t code = 0;
    int32_t r = pdf_document_has_timestamp(_handle, &code);
    if (r < 0 && error)
        *error = POXMakeError(code, @"hasTimestamp");
    return r;
}
- (POXDss*)dssWithError:(NSError**)error {
    int32_t code = 0;
    void* h = pdf_document_get_dss(_handle, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"dss");
        return nil;
    }
    return [[POXDss alloc] initWithHandle:h];
}

- (void)close {
    if (_handle) {
        pdf_document_free(_handle);
        _handle = NULL;
    }
}

@end

@implementation POXPdf {
    Pdf* _handle;
}

+ (instancetype)fromMarkdown:(NSString*)markdown error:(NSError**)error {
    int32_t code = 0;
    Pdf* h = pdf_from_markdown(markdown.UTF8String, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"fromMarkdown");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}
+ (instancetype)fromHtml:(NSString*)html error:(NSError**)error {
    int32_t code = 0;
    Pdf* h = pdf_from_html(html.UTF8String, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"fromHtml");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}
+ (instancetype)fromText:(NSString*)text error:(NSError**)error {
    int32_t code = 0;
    Pdf* h = pdf_from_text(text.UTF8String, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"fromText");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

// ── Phase-7: image / HTML+CSS constructors ───────────────────────────────────

+ (instancetype)fromImage:(NSString*)path error:(NSError**)error {
    int32_t code = 0;
    Pdf* h = pdf_from_image(path.UTF8String, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"fromImage");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

+ (instancetype)fromImageBytes:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    Pdf* h = pdf_from_image_bytes(data.bytes, (int32_t)data.length, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"fromImageBytes");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

+ (instancetype)fromHtml:(NSString*)html
                     css:(NSString*)css
               fontBytes:(NSData*)fontBytes
                   error:(NSError**)error {
    int32_t code = 0;
    Pdf* h = pdf_from_html_css(html.UTF8String, css.UTF8String,
                               fontBytes ? fontBytes.bytes : NULL,
                               fontBytes ? (uintptr_t)fontBytes.length : 0, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"fromHtmlCss");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

+ (instancetype)fromHtml:(NSString*)html
                     css:(NSString*)css
                families:(NSArray<NSString*>*)families
                   fonts:(NSArray<NSData*>*)fonts
                   error:(NSError**)error {
    NSUInteger count = families.count;
    const char** fams = count ? (const char**)malloc(sizeof(char*) * count) : NULL;
    const uint8_t** ptrs =
        count ? (const uint8_t**)malloc(sizeof(uint8_t*) * count) : NULL;
    uintptr_t* lens = count ? (uintptr_t*)malloc(sizeof(uintptr_t) * count) : NULL;
    for (NSUInteger i = 0; i < count; ++i) {
        fams[i] = families[i].UTF8String;
        NSData* f = i < fonts.count ? fonts[i] : [NSData data];
        ptrs[i] = (const uint8_t*)f.bytes;
        lens[i] = (uintptr_t)f.length;
    }
    int32_t code = 0;
    Pdf* h = pdf_from_html_css_with_fonts(html.UTF8String, css.UTF8String, fams, ptrs,
                                          lens, (uintptr_t)count, &code);
    if (fams)
        free(fams);
    if (ptrs)
        free(ptrs);
    if (lens)
        free(lens);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"fromHtmlCssWithFonts");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

- (instancetype)initWithHandle:(Pdf*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (void)dealloc {
    if (_handle)
        pdf_free(_handle);
}

- (BOOL)saveToPath:(NSString*)path error:(NSError**)error {
    int32_t code = 0;
    if (pdf_save(_handle, path.UTF8String, &code) != 0) {
        if (error)
            *error = POXMakeError(code, @"save");
        return NO;
    }
    return YES;
}

- (NSData*)toBytesWithError:(NSError**)error {
    int32_t len = 0, code = 0;
    uint8_t* p = pdf_save_to_bytes(_handle, &len, &code);
    if (!p) {
        if (error)
            *error = POXMakeError(code, @"saveToBytes");
        return nil;
    }
    NSData* out = [NSData dataWithBytes:p length:(len < 0 ? 0 : (NSUInteger)len)];
    free_bytes(p);
    return out;
}

- (NSInteger)pageCountError:(NSError**)error {
    int32_t code = 0;
    int32_t n = pdf_get_page_count(_handle, &code);
    if (n < 0) {
        if (error)
            *error = POXMakeError(code, @"pageCount");
        return -1;
    }
    return n;
}

- (void)close {
    if (_handle) {
        pdf_free(_handle);
        _handle = NULL;
    }
}

@end

@implementation POXPage {
    POXDocument* _document; // strong ref keeps the document alive
    NSInteger _index;
}

- (instancetype)initWithDocument:(POXDocument*)document index:(NSInteger)index {
    if ((self = [super init])) {
        _document = document;
        _index = index;
    }
    return self;
}

- (NSString*)text:(NSError**)error {
    return [_document extractText:_index error:error];
}
- (NSString*)markdown:(NSError**)error {
    return [_document toMarkdown:_index error:error];
}
- (NSString*)html:(NSError**)error {
    return [_document toHtml:_index error:error];
}
- (NSString*)plainText:(NSError**)error {
    return [_document toPlainText:_index error:error];
}

@end

// Copy a byte buffer return into NSData and free it via free_bytes. A null
// pointer is treated as a failure (sets `error`).
static NSData* _Nullable POXTakeBytes(uint8_t* p, NSUInteger len, int32_t code,
                                      NSString* op, NSError** error) {
    if (p == NULL) {
        if (error)
            *error = POXMakeError(code, op);
        return nil;
    }
    NSData* out = [NSData dataWithBytes:p length:len];
    free_bytes(p);
    return out;
}

@implementation POXDocumentEditor {
    DocumentEditor* _handle;
}

+ (instancetype)openEditor:(NSString*)path error:(NSError**)error {
    int32_t code = 0;
    DocumentEditor* h = document_editor_open(path.UTF8String, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"openEditor");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

+ (instancetype)openFromBytes:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    DocumentEditor* h = document_editor_open_from_bytes(data.bytes, data.length, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"openFromBytes");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

- (instancetype)initWithHandle:(DocumentEditor*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (void)dealloc {
    if (_handle)
        document_editor_free(_handle);
}

- (void)close {
    if (_handle) {
        document_editor_free(_handle);
        _handle = NULL;
    }
}

- (NSInteger)pageCountError:(NSError**)error {
    int32_t code = 0;
    int32_t n = document_editor_get_page_count(_handle, &code);
    if (n < 0) {
        if (error)
            *error = POXMakeError(code, @"pageCount");
        return -1;
    }
    return n;
}

- (POXVersion)version {
    POXVersion v = {0, 0};
    document_editor_get_version(_handle, &v.major, &v.minor);
    return v;
}

- (BOOL)isModified {
    return document_editor_is_modified(_handle) ? YES : NO;
}

- (NSString*)sourcePathError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(document_editor_get_source_path(_handle, &code), code,
                         @"sourcePath", error);
}

- (NSString*)producerError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(document_editor_get_producer(_handle, &code), code,
                         @"producer", error);
}

- (BOOL)setProducer:(NSString*)value error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_set_producer(_handle, value.UTF8String, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"setProducer");
        return NO;
    }
    return YES;
}

- (NSString*)creationDateError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(document_editor_get_creation_date(_handle, &code), code,
                         @"creationDate", error);
}

- (BOOL)setCreationDate:(NSString*)date error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_set_creation_date(_handle, date.UTF8String, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"setCreationDate");
        return NO;
    }
    return YES;
}

- (BOOL)deletePage:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_delete_page(_handle, (int32_t)page, &code) != 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, @"deletePage");
        return NO;
    }
    return YES;
}

- (BOOL)movePageFrom:(NSInteger)from to:(NSInteger)to error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_move_page(_handle, (int32_t)from, (int32_t)to, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"movePage");
        return NO;
    }
    return YES;
}

- (BOOL)rotatePage:(NSInteger)page byDegrees:(NSInteger)degrees error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_rotate_page_by(_handle, (uintptr_t)page, (int32_t)degrees,
                                       &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"rotatePageBy");
        return NO;
    }
    return YES;
}

- (BOOL)rotateAllPages:(NSInteger)degrees error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_rotate_all_pages(_handle, (int32_t)degrees, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"rotateAllPages");
        return NO;
    }
    return YES;
}

- (BOOL)setPageRotation:(NSInteger)page
                degrees:(NSInteger)degrees
                  error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_set_page_rotation(_handle, (int32_t)page, (int32_t)degrees,
                                          &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"setPageRotation");
        return NO;
    }
    return YES;
}

- (NSInteger)pageRotation:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    int32_t r = document_editor_get_page_rotation(_handle, (int32_t)page, &code);
    if (r < 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, @"getPageRotation");
        return -1;
    }
    return r;
}

- (BOOL)cropMarginsLeft:(float)left
                  right:(float)right
                    top:(float)top
                 bottom:(float)bottom
                  error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_crop_margins(_handle, left, right, top, bottom, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"cropMargins");
        return NO;
    }
    return YES;
}

- (POXBbox)pageCropBox:(NSInteger)page error:(NSError**)error {
    POXBbox box = {0, 0, 0, 0};
    int32_t code = 0;
    double x = 0, y = 0, w = 0, h = 0;
    if (document_editor_get_page_crop_box(_handle, (uintptr_t)page, &x, &y, &w, &h,
                                          &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"getPageCropBox");
        return box;
    }
    box.x = (float)x;
    box.y = (float)y;
    box.width = (float)w;
    box.height = (float)h;
    return box;
}

- (BOOL)setPageCropBox:(NSInteger)page box:(POXBbox)box error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_set_page_crop_box(_handle, (uintptr_t)page, box.x, box.y,
                                          box.width, box.height, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"setPageCropBox");
        return NO;
    }
    return YES;
}

- (POXBbox)pageMediaBox:(NSInteger)page error:(NSError**)error {
    POXBbox box = {0, 0, 0, 0};
    int32_t code = 0;
    double x = 0, y = 0, w = 0, h = 0;
    if (document_editor_get_page_media_box(_handle, (uintptr_t)page, &x, &y, &w, &h,
                                           &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"getPageMediaBox");
        return box;
    }
    box.x = (float)x;
    box.y = (float)y;
    box.width = (float)w;
    box.height = (float)h;
    return box;
}

- (BOOL)setPageMediaBox:(NSInteger)page box:(POXBbox)box error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_set_page_media_box(_handle, (uintptr_t)page, box.x, box.y,
                                           box.width, box.height, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"setPageMediaBox");
        return NO;
    }
    return YES;
}

- (BOOL)applyAllRedactions:(NSError**)error {
    int32_t code = 0;
    if (document_editor_apply_all_redactions(_handle, &code) != 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, @"applyAllRedactions");
        return NO;
    }
    return YES;
}

- (BOOL)applyPageRedactions:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_apply_page_redactions(_handle, (uintptr_t)page, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"applyPageRedactions");
        return NO;
    }
    return YES;
}

- (BOOL)isPageMarkedForRedaction:(NSInteger)page {
    return document_editor_is_page_marked_for_redaction(_handle, (uintptr_t)page) == 1
               ? YES
               : NO;
}

- (BOOL)unmarkPageForRedaction:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_unmark_page_for_redaction(_handle, (uintptr_t)page, &code) !=
            0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"unmarkPageForRedaction");
        return NO;
    }
    return YES;
}

- (BOOL)eraseRegion:(NSInteger)page
                  x:(float)x
                  y:(float)y
                  w:(float)w
                  h:(float)h
              error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_erase_region(_handle, (int32_t)page, x, y, w, h, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"eraseRegion");
        return NO;
    }
    return YES;
}

- (BOOL)eraseRegions:(NSInteger)page
               rects:(NSArray<NSValue*>*)rects
               error:(NSError**)error {
    NSUInteger count = rects.count;
    double* flat = count ? (double*)malloc(sizeof(double) * count * 4) : NULL;
    for (NSUInteger i = 0; i < count; ++i) {
        POXBbox b = {0, 0, 0, 0};
        [rects[i] getValue:&b];
        flat[i * 4 + 0] = (double)b.x;
        flat[i * 4 + 1] = (double)b.y;
        flat[i * 4 + 2] = (double)b.width;
        flat[i * 4 + 3] = (double)b.height;
    }
    int32_t code = 0;
    int32_t rc = document_editor_erase_regions(_handle, (uintptr_t)page, flat,
                                               (uintptr_t)count, &code);
    if (flat)
        free(flat);
    if (rc != 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, @"eraseRegions");
        return NO;
    }
    return YES;
}

- (BOOL)clearEraseRegions:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_clear_erase_regions(_handle, (uintptr_t)page, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"clearEraseRegions");
        return NO;
    }
    return YES;
}

- (BOOL)flattenForms:(NSError**)error {
    int32_t code = 0;
    if (document_editor_flatten_forms(_handle, &code) != 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, @"flattenForms");
        return NO;
    }
    return YES;
}

- (BOOL)flattenFormsOnPage:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_flatten_forms_on_page(_handle, (int32_t)page, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"flattenFormsOnPage");
        return NO;
    }
    return YES;
}

- (BOOL)flattenAnnotations:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_flatten_annotations(_handle, (int32_t)page, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"flattenAnnotations");
        return NO;
    }
    return YES;
}

- (BOOL)flattenAllAnnotations:(NSError**)error {
    int32_t code = 0;
    if (document_editor_flatten_all_annotations(_handle, &code) != 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, @"flattenAllAnnotations");
        return NO;
    }
    return YES;
}

- (NSInteger)flattenWarningsCount {
    return document_editor_flatten_warnings_count(_handle);
}

- (NSString*)flattenWarning:(NSInteger)index error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(
        document_editor_flatten_warning(_handle, (int32_t)index, &code), code,
        @"flattenWarning", error);
}

- (BOOL)isPageMarkedForFlatten:(NSInteger)page {
    return document_editor_is_page_marked_for_flatten(_handle, (uintptr_t)page) == 1
               ? YES
               : NO;
}

- (BOOL)unmarkPageForFlatten:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_unmark_page_for_flatten(_handle, (uintptr_t)page, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"unmarkPageForFlatten");
        return NO;
    }
    return YES;
}

- (BOOL)setFormField:(NSString*)name value:(NSString*)value error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_set_form_field_value(_handle, name.UTF8String, value.UTF8String,
                                             &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"setFormFieldValue");
        return NO;
    }
    return YES;
}

- (BOOL)mergeFrom:(NSString*)sourcePath error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_merge_from(_handle, sourcePath.UTF8String, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"mergeFrom");
        return NO;
    }
    return YES;
}

- (BOOL)mergeFromBytes:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_merge_from_bytes(_handle, data.bytes, data.length, &code) !=
            0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"mergeFromBytes");
        return NO;
    }
    return YES;
}

- (BOOL)convertToPdfA:(NSInteger)level error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_convert_to_pdf_a(_handle, (int32_t)level, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"convertToPdfA");
        return NO;
    }
    return YES;
}

- (BOOL)embedFile:(NSString*)name data:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_embed_file(_handle, name.UTF8String, data.bytes, data.length,
                                   &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"embedFile");
        return NO;
    }
    return YES;
}

- (NSData*)extractPagesToBytes:(NSArray<NSNumber*>*)pages error:(NSError**)error {
    NSUInteger count = pages.count;
    int32_t* idx = count ? (int32_t*)malloc(sizeof(int32_t) * count) : NULL;
    for (NSUInteger i = 0; i < count; ++i)
        idx[i] = (int32_t)[pages[i] integerValue];
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = document_editor_extract_pages_to_bytes(_handle, idx, (uintptr_t)count,
                                                        &len, &code);
    if (idx)
        free(idx);
    return POXTakeBytes(p, (NSUInteger)len, code, @"extractPagesToBytes", error);
}

- (BOOL)saveToPath:(NSString*)path error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_save(_handle, path.UTF8String, &code) != 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, @"save");
        return NO;
    }
    return YES;
}

- (NSData*)saveToBytesWithError:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = document_editor_save_to_bytes(_handle, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"saveToBytes", error);
}

- (NSData*)saveToBytesCompress:(BOOL)compress
                garbageCollect:(BOOL)garbageCollect
                     linearize:(BOOL)linearize
                         error:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = document_editor_save_to_bytes_with_options(
        _handle, compress ? true : false, garbageCollect ? true : false,
        linearize ? true : false, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"saveToBytesWithOptions", error);
}

- (BOOL)saveEncryptedToPath:(NSString*)path
               userPassword:(NSString*)userPassword
              ownerPassword:(NSString*)ownerPassword
                      error:(NSError**)error {
    int32_t code = 0;
    if (document_editor_save_encrypted(_handle, path.UTF8String,
                                       userPassword.UTF8String,
                                       ownerPassword.UTF8String, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"saveEncrypted");
        return NO;
    }
    return YES;
}

- (NSData*)saveEncryptedToBytesWithUserPassword:(NSString*)userPassword
                                  ownerPassword:(NSString*)ownerPassword
                                          error:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = document_editor_save_encrypted_to_bytes(
        _handle, userPassword.UTF8String, ownerPassword.UTF8String, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"saveEncryptedToBytes", error);
}

// ── Phase-7: geometric redaction + barcode placement ─────────────────────────

- (BOOL)redactionAddPage:(NSInteger)page
                      x1:(double)x1
                      y1:(double)y1
                      x2:(double)x2
                      y2:(double)y2
                       r:(double)r
                       g:(double)g
                       b:(double)b
                   error:(NSError**)error {
    int32_t code = 0;
    if (pdf_redaction_add(_handle, (uintptr_t)page, x1, y1, x2, y2, r, g, b, &code) !=
            0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"redactionAdd");
        return NO;
    }
    return YES;
}

- (int32_t)redactionCount:(NSInteger)page error:(NSError**)error {
    int32_t code = 0;
    int32_t n = pdf_redaction_count(_handle, (uintptr_t)page, &code);
    if (n < 0 && error)
        *error = POXMakeError(code, @"redactionCount");
    return n;
}

- (int32_t)redactionApplyScrubMetadata:(BOOL)scrubMetadata
                                     r:(double)r
                                     g:(double)g
                                     b:(double)b
                                 error:(NSError**)error {
    int32_t code = 0;
    int32_t n =
        pdf_redaction_apply(_handle, scrubMetadata ? true : false, r, g, b, &code);
    if (n < 0 && error)
        *error = POXMakeError(code, @"redactionApply");
    return n;
}

- (int32_t)redactionScrubMetadataWithError:(NSError**)error {
    int32_t code = 0;
    int32_t n = pdf_redaction_scrub_metadata(_handle, &code);
    if (n < 0 && error)
        *error = POXMakeError(code, @"redactionScrubMetadata");
    return n;
}

- (BOOL)addBarcode:(POXBarcode*)barcode
              page:(NSInteger)page
                 x:(float)x
                 y:(float)y
             width:(float)width
            height:(float)height
             error:(NSError**)error {
    int32_t code = 0;
    if (pdf_add_barcode_to_page(_handle, (int32_t)page, [barcode POX_handle], x, y,
                                width, height, &code) != 0 ||
        code != 0) {
        if (error)
            *error = POXMakeError(code, @"addBarcodeToPage");
        return NO;
    }
    return YES;
}

- (BOOL)importFdfBytes:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    int32_t rc =
        pdf_editor_import_fdf_bytes(_handle, data.bytes, (uintptr_t)data.length, &code);
    if (rc != 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, @"importFdfBytes");
        return NO;
    }
    return YES;
}

- (BOOL)importXfdfBytes:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    int32_t rc = pdf_editor_import_xfdf_bytes(_handle, data.bytes,
                                              (uintptr_t)data.length, &code);
    if (rc != 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, @"importXfdfBytes");
        return NO;
    }
    return YES;
}

@end

// ── PDF creation builder API ─────────────────────────────────────────────────
//
// POXDocumentBuilder owns an FfiDocumentBuilder; -page:/-letterPage/-a4Page
// produce a POXPageBuilder owning an FfiPageBuilder. POXEmbeddedFont owns an
// EmbeddedFont, whose ownership transfers to the builder on a successful
// register (the wrapper handle is then nulled). All int32 ops are status codes
// (0 == ok); a non-zero return or non-zero error_code surfaces as NSError.

// Private initializers used across builder types.
@interface POXEmbeddedFont ()
- (instancetype)initWithHandle:(EmbeddedFont*)handle;
- (EmbeddedFont*)takeHandle; // releases ownership without freeing
@end

@interface POXPageBuilder ()
- (instancetype)initWithHandle:(FfiPageBuilder*)handle;
@end

@implementation POXEmbeddedFont {
    EmbeddedFont* _handle;
}

+ (instancetype)fromPath:(NSString*)path error:(NSError**)error {
    int32_t code = 0;
    EmbeddedFont* h = pdf_embedded_font_from_file(path.UTF8String, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"embeddedFontFromFile");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

+ (instancetype)fromBytes:(NSData*)data name:(NSString*)name error:(NSError**)error {
    int32_t code = 0;
    EmbeddedFont* h = pdf_embedded_font_from_bytes(
        data.bytes, data.length, name ? name.UTF8String : NULL, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"embeddedFontFromBytes");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

- (instancetype)initWithHandle:(EmbeddedFont*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (EmbeddedFont*)takeHandle {
    EmbeddedFont* h = _handle;
    _handle = NULL;
    return h;
}

- (void)close {
    if (_handle) {
        pdf_embedded_font_free(_handle);
        _handle = NULL;
    }
}

- (void)dealloc {
    if (_handle)
        pdf_embedded_font_free(_handle);
}

@end

@implementation POXPageBuilder {
    FfiPageBuilder* _handle;
}

- (instancetype)initWithHandle:(FfiPageBuilder*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (void)dealloc {
    if (_handle)
        pdf_page_builder_free(_handle);
}

// Guard against use after -done/-close. Returns NO and sets `error` if closed.
- (BOOL)checkOpen:(NSString*)op error:(NSError**)error {
    if (!_handle) {
        if (error)
            *error = POXMakeError(0, op);
        return NO;
    }
    return YES;
}

// Wrap a status-code (0 == ok) FFI op result + error_code into BOOL + NSError.
static BOOL POXPageStatus(int32_t rc, int32_t code, NSString* op, NSError** error) {
    if (rc != 0 || code != 0) {
        if (error)
            *error = POXMakeError(code, op);
        return NO;
    }
    return YES;
}

#define POX_PAGE_GUARD(op)                                                             \
    if (![self checkOpen:(op) error:error])                                            \
    return NO

- (BOOL)font:(NSString*)name size:(float)size error:(NSError**)error {
    POX_PAGE_GUARD(@"pageFont");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_font(_handle, name.UTF8String, size, &code),
                         code, @"pageFont", error);
}

- (BOOL)at:(float)x y:(float)y error:(NSError**)error {
    POX_PAGE_GUARD(@"pageAt");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_at(_handle, x, y, &code), code, @"pageAt",
                         error);
}

- (BOOL)text:(NSString*)text error:(NSError**)error {
    POX_PAGE_GUARD(@"pageText");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_text(_handle, text.UTF8String, &code), code,
                         @"pageText", error);
}

- (BOOL)heading:(uint8_t)level text:(NSString*)text error:(NSError**)error {
    POX_PAGE_GUARD(@"pageHeading");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_heading(_handle, level, text.UTF8String, &code), code,
        @"pageHeading", error);
}

- (BOOL)paragraph:(NSString*)text error:(NSError**)error {
    POX_PAGE_GUARD(@"pageParagraph");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_paragraph(_handle, text.UTF8String, &code),
                         code, @"pageParagraph", error);
}

- (BOOL)space:(float)points error:(NSError**)error {
    POX_PAGE_GUARD(@"pageSpace");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_space(_handle, points, &code), code,
                         @"pageSpace", error);
}

- (BOOL)horizontalRule:(NSError**)error {
    POX_PAGE_GUARD(@"pageHorizontalRule");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_horizontal_rule(_handle, &code), code,
                         @"pageHorizontalRule", error);
}

- (BOOL)linkUrl:(NSString*)url error:(NSError**)error {
    POX_PAGE_GUARD(@"pageLinkUrl");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_link_url(_handle, url.UTF8String, &code),
                         code, @"pageLinkUrl", error);
}

- (BOOL)linkPage:(NSInteger)page error:(NSError**)error {
    POX_PAGE_GUARD(@"pageLinkPage");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_link_page(_handle, (uintptr_t)page, &code),
                         code, @"pageLinkPage", error);
}

- (BOOL)linkNamed:(NSString*)destination error:(NSError**)error {
    POX_PAGE_GUARD(@"pageLinkNamed");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_link_named(_handle, destination.UTF8String, &code), code,
        @"pageLinkNamed", error);
}

- (BOOL)linkJavascript:(NSString*)script error:(NSError**)error {
    POX_PAGE_GUARD(@"pageLinkJavascript");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_link_javascript(_handle, script.UTF8String, &code), code,
        @"pageLinkJavascript", error);
}

- (BOOL)onOpen:(NSString*)script error:(NSError**)error {
    POX_PAGE_GUARD(@"pageOnOpen");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_on_open(_handle, script.UTF8String, &code),
                         code, @"pageOnOpen", error);
}

- (BOOL)onClose:(NSString*)script error:(NSError**)error {
    POX_PAGE_GUARD(@"pageOnClose");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_on_close(_handle, script.UTF8String, &code),
                         code, @"pageOnClose", error);
}

- (BOOL)fieldKeystroke:(NSString*)script error:(NSError**)error {
    POX_PAGE_GUARD(@"pageFieldKeystroke");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_field_keystroke(_handle, script.UTF8String, &code), code,
        @"pageFieldKeystroke", error);
}

- (BOOL)fieldFormat:(NSString*)script error:(NSError**)error {
    POX_PAGE_GUARD(@"pageFieldFormat");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_field_format(_handle, script.UTF8String, &code), code,
        @"pageFieldFormat", error);
}

- (BOOL)fieldValidate:(NSString*)script error:(NSError**)error {
    POX_PAGE_GUARD(@"pageFieldValidate");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_field_validate(_handle, script.UTF8String, &code), code,
        @"pageFieldValidate", error);
}

- (BOOL)fieldCalculate:(NSString*)script error:(NSError**)error {
    POX_PAGE_GUARD(@"pageFieldCalculate");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_field_calculate(_handle, script.UTF8String, &code), code,
        @"pageFieldCalculate", error);
}

- (BOOL)highlightR:(float)r g:(float)g b:(float)b error:(NSError**)error {
    POX_PAGE_GUARD(@"pageHighlight");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_highlight(_handle, r, g, b, &code), code,
                         @"pageHighlight", error);
}

- (BOOL)underlineR:(float)r g:(float)g b:(float)b error:(NSError**)error {
    POX_PAGE_GUARD(@"pageUnderline");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_underline(_handle, r, g, b, &code), code,
                         @"pageUnderline", error);
}

- (BOOL)strikeoutR:(float)r g:(float)g b:(float)b error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStrikeout");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_strikeout(_handle, r, g, b, &code), code,
                         @"pageStrikeout", error);
}

- (BOOL)squigglyR:(float)r g:(float)g b:(float)b error:(NSError**)error {
    POX_PAGE_GUARD(@"pageSquiggly");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_squiggly(_handle, r, g, b, &code), code,
                         @"pageSquiggly", error);
}

- (BOOL)stickyNote:(NSString*)text error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStickyNote");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_sticky_note(_handle, text.UTF8String, &code),
                         code, @"pageStickyNote", error);
}

- (BOOL)stickyNoteAt:(float)x y:(float)y text:(NSString*)text error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStickyNoteAt");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_sticky_note_at(_handle, x, y, text.UTF8String, &code), code,
        @"pageStickyNoteAt", error);
}

- (BOOL)watermark:(NSString*)text error:(NSError**)error {
    POX_PAGE_GUARD(@"pageWatermark");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_watermark(_handle, text.UTF8String, &code),
                         code, @"pageWatermark", error);
}

- (BOOL)watermarkConfidential:(NSError**)error {
    POX_PAGE_GUARD(@"pageWatermarkConfidential");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_watermark_confidential(_handle, &code), code,
                         @"pageWatermarkConfidential", error);
}

- (BOOL)watermarkDraft:(NSError**)error {
    POX_PAGE_GUARD(@"pageWatermarkDraft");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_watermark_draft(_handle, &code), code,
                         @"pageWatermarkDraft", error);
}

- (BOOL)stamp:(NSString*)typeName error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStamp");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_stamp(_handle, typeName.UTF8String, &code),
                         code, @"pageStamp", error);
}

- (BOOL)freetextAtX:(float)x
                  y:(float)y
                  w:(float)w
                  h:(float)h
               text:(NSString*)text
              error:(NSError**)error {
    POX_PAGE_GUARD(@"pageFreetext");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_freetext(_handle, x, y, w, h, text.UTF8String, &code), code,
        @"pageFreetext", error);
}

- (BOOL)textFieldName:(NSString*)name
                    x:(float)x
                    y:(float)y
                    w:(float)w
                    h:(float)h
         defaultValue:(NSString*)defaultValue
                error:(NSError**)error {
    POX_PAGE_GUARD(@"pageTextField");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_text_field(
                             _handle, name.UTF8String, x, y, w, h,
                             defaultValue ? defaultValue.UTF8String : NULL, &code),
                         code, @"pageTextField", error);
}

- (BOOL)checkboxName:(NSString*)name
                   x:(float)x
                   y:(float)y
                   w:(float)w
                   h:(float)h
             checked:(BOOL)checked
               error:(NSError**)error {
    POX_PAGE_GUARD(@"pageCheckbox");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_checkbox(_handle, name.UTF8String, x, y, w, h,
                                                   checked ? 1 : 0, &code),
                         code, @"pageCheckbox", error);
}

- (BOOL)comboBoxName:(NSString*)name
                   x:(float)x
                   y:(float)y
                   w:(float)w
                   h:(float)h
             options:(NSArray<NSString*>*)options
            selected:(NSString*)selected
               error:(NSError**)error {
    POX_PAGE_GUARD(@"pageComboBox");
    NSUInteger count = options.count;
    const char** opts = count ? (const char**)malloc(sizeof(char*) * count) : NULL;
    for (NSUInteger i = 0; i < count; ++i)
        opts[i] = options[i].UTF8String;
    int32_t code = 0;
    int32_t rc = pdf_page_builder_combo_box(
        _handle, name.UTF8String, x, y, w, h, opts, (uintptr_t)count,
        selected ? selected.UTF8String : NULL, &code);
    if (opts)
        free(opts);
    return POXPageStatus(rc, code, @"pageComboBox", error);
}

- (BOOL)radioGroupName:(NSString*)name
                values:(NSArray<NSString*>*)values
                    xs:(NSArray<NSNumber*>*)xs
                    ys:(NSArray<NSNumber*>*)ys
                    ws:(NSArray<NSNumber*>*)ws
                    hs:(NSArray<NSNumber*>*)hs
              selected:(NSString*)selected
                 error:(NSError**)error {
    POX_PAGE_GUARD(@"pageRadioGroup");
    NSUInteger count = values.count;
    const char** vals = count ? (const char**)malloc(sizeof(char*) * count) : NULL;
    float* fxs = count ? (float*)malloc(sizeof(float) * count) : NULL;
    float* fys = count ? (float*)malloc(sizeof(float) * count) : NULL;
    float* fws = count ? (float*)malloc(sizeof(float) * count) : NULL;
    float* fhs = count ? (float*)malloc(sizeof(float) * count) : NULL;
    for (NSUInteger i = 0; i < count; ++i) {
        vals[i] = values[i].UTF8String;
        fxs[i] = i < xs.count ? xs[i].floatValue : 0.0f;
        fys[i] = i < ys.count ? ys[i].floatValue : 0.0f;
        fws[i] = i < ws.count ? ws[i].floatValue : 0.0f;
        fhs[i] = i < hs.count ? hs[i].floatValue : 0.0f;
    }
    int32_t code = 0;
    int32_t rc = pdf_page_builder_radio_group(
        _handle, name.UTF8String, vals, fxs, fys, fws, fhs, (uintptr_t)count,
        selected ? selected.UTF8String : NULL, &code);
    if (vals)
        free(vals);
    if (fxs)
        free(fxs);
    if (fys)
        free(fys);
    if (fws)
        free(fws);
    if (fhs)
        free(fhs);
    return POXPageStatus(rc, code, @"pageRadioGroup", error);
}

- (BOOL)pushButtonName:(NSString*)name
                     x:(float)x
                     y:(float)y
                     w:(float)w
                     h:(float)h
               caption:(NSString*)caption
                 error:(NSError**)error {
    POX_PAGE_GUARD(@"pagePushButton");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_push_button(_handle, name.UTF8String, x, y, w,
                                                      h, caption.UTF8String, &code),
                         code, @"pagePushButton", error);
}

- (BOOL)signatureFieldName:(NSString*)name
                         x:(float)x
                         y:(float)y
                         w:(float)w
                         h:(float)h
                     error:(NSError**)error {
    POX_PAGE_GUARD(@"pageSignatureField");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_signature_field(_handle, name.UTF8String, x, y, w, h, &code),
        code, @"pageSignatureField", error);
}

- (BOOL)footnoteRefMark:(NSString*)refMark
               noteText:(NSString*)noteText
                  error:(NSError**)error {
    POX_PAGE_GUARD(@"pageFootnote");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_footnote(_handle, refMark.UTF8String,
                                                   noteText.UTF8String, &code),
                         code, @"pageFootnote", error);
}

- (BOOL)columnsCount:(uint32_t)columnCount
               gapPt:(float)gapPt
                text:(NSString*)text
               error:(NSError**)error {
    POX_PAGE_GUARD(@"pageColumns");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_columns(_handle, columnCount, gapPt, text.UTF8String, &code),
        code, @"pageColumns", error);
}

- (BOOL)inlineText:(NSString*)text error:(NSError**)error {
    POX_PAGE_GUARD(@"pageInline");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_inline(_handle, text.UTF8String, &code), code,
                         @"pageInline", error);
}

- (BOOL)inlineBold:(NSString*)text error:(NSError**)error {
    POX_PAGE_GUARD(@"pageInlineBold");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_inline_bold(_handle, text.UTF8String, &code),
                         code, @"pageInlineBold", error);
}

- (BOOL)inlineItalic:(NSString*)text error:(NSError**)error {
    POX_PAGE_GUARD(@"pageInlineItalic");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_inline_italic(_handle, text.UTF8String, &code), code,
        @"pageInlineItalic", error);
}

- (BOOL)inlineColorR:(float)r
                   g:(float)g
                   b:(float)b
                text:(NSString*)text
               error:(NSError**)error {
    POX_PAGE_GUARD(@"pageInlineColor");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_inline_color(_handle, r, g, b, text.UTF8String, &code), code,
        @"pageInlineColor", error);
}

- (BOOL)newline:(NSError**)error {
    POX_PAGE_GUARD(@"pageNewline");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_newline(_handle, &code), code, @"pageNewline",
                         error);
}

- (BOOL)barcode1d:(int32_t)barcodeType
             data:(NSString*)data
                x:(float)x
                y:(float)y
                w:(float)w
                h:(float)h
            error:(NSError**)error {
    POX_PAGE_GUARD(@"pageBarcode1d");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_barcode_1d(
                             _handle, barcodeType, data.UTF8String, x, y, w, h, &code),
                         code, @"pageBarcode1d", error);
}

- (BOOL)barcodeQrData:(NSString*)data
                    x:(float)x
                    y:(float)y
                 size:(float)size
                error:(NSError**)error {
    POX_PAGE_GUARD(@"pageBarcodeQr");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_barcode_qr(_handle, data.UTF8String, x, y, size, &code), code,
        @"pageBarcodeQr", error);
}

- (BOOL)image:(NSData*)bytes
            x:(float)x
            y:(float)y
            w:(float)w
            h:(float)h
        error:(NSError**)error {
    POX_PAGE_GUARD(@"pageImage");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_image(_handle, bytes.bytes, bytes.length, x, y, w, h, &code),
        code, @"pageImage", error);
}

- (BOOL)imageWithAlt:(NSData*)bytes
                   x:(float)x
                   y:(float)y
                   w:(float)w
                   h:(float)h
             altText:(NSString*)altText
               error:(NSError**)error {
    POX_PAGE_GUARD(@"pageImageWithAlt");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_image_with_alt(_handle, bytes.bytes,
                                                         bytes.length, x, y, w, h,
                                                         altText.UTF8String, &code),
                         code, @"pageImageWithAlt", error);
}

- (BOOL)imageArtifact:(NSData*)bytes
                    x:(float)x
                    y:(float)y
                    w:(float)w
                    h:(float)h
                error:(NSError**)error {
    POX_PAGE_GUARD(@"pageImageArtifact");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_image_artifact(
                             _handle, bytes.bytes, bytes.length, x, y, w, h, &code),
                         code, @"pageImageArtifact", error);
}

- (BOOL)rectX:(float)x y:(float)y w:(float)w h:(float)h error:(NSError**)error {
    POX_PAGE_GUARD(@"pageRect");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_rect(_handle, x, y, w, h, &code), code,
                         @"pageRect", error);
}

- (BOOL)filledRectX:(float)x
                  y:(float)y
                  w:(float)w
                  h:(float)h
                  r:(float)r
                  g:(float)g
                  b:(float)b
              error:(NSError**)error {
    POX_PAGE_GUARD(@"pageFilledRect");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_filled_rect(_handle, x, y, w, h, r, g, b, &code), code,
        @"pageFilledRect", error);
}

- (BOOL)lineX1:(float)x1 y1:(float)y1 x2:(float)x2 y2:(float)y2 error:(NSError**)error {
    POX_PAGE_GUARD(@"pageLine");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_line(_handle, x1, y1, x2, y2, &code), code,
                         @"pageLine", error);
}

- (BOOL)strokeRectX:(float)x
                  y:(float)y
                  w:(float)w
                  h:(float)h
              width:(float)width
                  r:(float)r
                  g:(float)g
                  b:(float)b
              error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStrokeRect");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_stroke_rect(_handle, x, y, w, h, width, r, g, b, &code), code,
        @"pageStrokeRect", error);
}

- (BOOL)strokeLineX1:(float)x1
                  y1:(float)y1
                  x2:(float)x2
                  y2:(float)y2
               width:(float)width
                   r:(float)r
                   g:(float)g
                   b:(float)b
               error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStrokeLine");
    int32_t code = 0;
    return POXPageStatus(
        pdf_page_builder_stroke_line(_handle, x1, y1, x2, y2, width, r, g, b, &code),
        code, @"pageStrokeLine", error);
}

// Marshal an NSArray<NSNumber*> dash pattern into a heap float array.
static float* POXFloatArray(NSArray<NSNumber*>* nums, NSUInteger* outCount) {
    NSUInteger n = nums.count;
    *outCount = n;
    if (n == 0)
        return NULL;
    float* a = (float*)malloc(sizeof(float) * n);
    for (NSUInteger i = 0; i < n; ++i)
        a[i] = nums[i].floatValue;
    return a;
}

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
                    error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStrokeRectDashed");
    NSUInteger n = 0;
    float* dash = POXFloatArray(dashArray, &n);
    int32_t code = 0;
    int32_t rc = pdf_page_builder_stroke_rect_dashed(
        _handle, x, y, w, h, width, r, g, b, dash, (uintptr_t)n, phase, &code);
    if (dash)
        free(dash);
    return POXPageStatus(rc, code, @"pageStrokeRectDashed", error);
}

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
                     error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStrokeLineDashed");
    NSUInteger n = 0;
    float* dash = POXFloatArray(dashArray, &n);
    int32_t code = 0;
    int32_t rc = pdf_page_builder_stroke_line_dashed(
        _handle, x1, y1, x2, y2, width, r, g, b, dash, (uintptr_t)n, phase, &code);
    if (dash)
        free(dash);
    return POXPageStatus(rc, code, @"pageStrokeLineDashed", error);
}

- (BOOL)textInRectX:(float)x
                  y:(float)y
                  w:(float)w
                  h:(float)h
               text:(NSString*)text
              align:(int32_t)align
              error:(NSError**)error {
    POX_PAGE_GUARD(@"pageTextInRect");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_text_in_rect(_handle, x, y, w, h,
                                                       text.UTF8String, align, &code),
                         code, @"pageTextInRect", error);
}

- (BOOL)newPageSameSize:(NSError**)error {
    POX_PAGE_GUARD(@"pageNewPageSameSize");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_new_page_same_size(_handle, &code), code,
                         @"pageNewPageSameSize", error);
}

- (BOOL)tableColumns:(NSArray<NSNumber*>*)widths
              aligns:(NSArray<NSNumber*>*)aligns
                rows:(NSArray<NSArray<NSString*>*>*)rows
           hasHeader:(BOOL)hasHeader
               error:(NSError**)error {
    POX_PAGE_GUARD(@"pageTable");
    NSUInteger nCols = widths.count;
    NSUInteger nRows = rows.count;
    NSUInteger nCells = nCols * nRows;
    float* fwidths = nCols ? (float*)malloc(sizeof(float) * nCols) : NULL;
    int32_t* ialigns = nCols ? (int32_t*)malloc(sizeof(int32_t) * nCols) : NULL;
    for (NSUInteger c = 0; c < nCols; ++c) {
        fwidths[c] = widths[c].floatValue;
        ialigns[c] = c < aligns.count ? (int32_t)aligns[c].integerValue : 0;
    }
    // Guard on (nRows && nCols), the exact conditions under which the fill
    // loops below run, so the analyzer can prove `cells` is non-NULL on that
    // path (matching fwidths/ialigns above). nCells == nCols * nRows.
    const char** cells =
        (nRows && nCols) ? (const char**)malloc(sizeof(char*) * nCells) : NULL;
    for (NSUInteger rIdx = 0; rIdx < nRows; ++rIdx) {
        NSArray<NSString*>* row = rows[rIdx];
        for (NSUInteger c = 0; c < nCols; ++c)
            cells[rIdx * nCols + c] = c < row.count ? row[c].UTF8String : "";
    }
    int32_t code = 0;
    int32_t rc =
        pdf_page_builder_table(_handle, (uintptr_t)nCols, fwidths, ialigns,
                               (uintptr_t)nRows, cells, hasHeader ? 1 : 0, &code);
    if (fwidths)
        free(fwidths);
    if (ialigns)
        free(ialigns);
    if (cells)
        free(cells);
    return POXPageStatus(rc, code, @"pageTable", error);
}

- (BOOL)streamingTableBeginHeaders:(NSArray<NSString*>*)headers
                            widths:(NSArray<NSNumber*>*)widths
                            aligns:(NSArray<NSNumber*>*)aligns
                      repeatHeader:(BOOL)repeatHeader
                             error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStreamingTableBegin");
    NSUInteger n = headers.count;
    const char** hdrs = n ? (const char**)malloc(sizeof(char*) * n) : NULL;
    float* fwidths = n ? (float*)malloc(sizeof(float) * n) : NULL;
    int32_t* ialigns = n ? (int32_t*)malloc(sizeof(int32_t) * n) : NULL;
    for (NSUInteger i = 0; i < n; ++i) {
        hdrs[i] = headers[i].UTF8String;
        fwidths[i] = i < widths.count ? widths[i].floatValue : 0.0f;
        ialigns[i] = i < aligns.count ? (int32_t)aligns[i].integerValue : 0;
    }
    int32_t code = 0;
    int32_t rc = pdf_page_builder_streaming_table_begin(
        _handle, (uintptr_t)n, hdrs, fwidths, ialigns, repeatHeader ? 1 : 0, &code);
    if (hdrs)
        free(hdrs);
    if (fwidths)
        free(fwidths);
    if (ialigns)
        free(ialigns);
    return POXPageStatus(rc, code, @"pageStreamingTableBegin", error);
}

- (BOOL)streamingTableBeginV2Headers:(NSArray<NSString*>*)headers
                              widths:(NSArray<NSNumber*>*)widths
                              aligns:(NSArray<NSNumber*>*)aligns
                        repeatHeader:(BOOL)repeatHeader
                                mode:(int32_t)mode
                          sampleRows:(NSInteger)sampleRows
                       minColWidthPt:(float)minColWidthPt
                       maxColWidthPt:(float)maxColWidthPt
                          maxRowspan:(NSInteger)maxRowspan
                               error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStreamingTableBeginV2");
    NSUInteger n = headers.count;
    const char** hdrs = n ? (const char**)malloc(sizeof(char*) * n) : NULL;
    float* fwidths = n ? (float*)malloc(sizeof(float) * n) : NULL;
    int32_t* ialigns = n ? (int32_t*)malloc(sizeof(int32_t) * n) : NULL;
    for (NSUInteger i = 0; i < n; ++i) {
        hdrs[i] = headers[i].UTF8String;
        fwidths[i] = i < widths.count ? widths[i].floatValue : 0.0f;
        ialigns[i] = i < aligns.count ? (int32_t)aligns[i].integerValue : 0;
    }
    int32_t code = 0;
    int32_t rc = pdf_page_builder_streaming_table_begin_v2(
        _handle, (uintptr_t)n, hdrs, fwidths, ialigns, repeatHeader ? 1 : 0, mode,
        (uintptr_t)sampleRows, minColWidthPt, maxColWidthPt, (uintptr_t)maxRowspan,
        &code);
    if (hdrs)
        free(hdrs);
    if (fwidths)
        free(fwidths);
    if (ialigns)
        free(ialigns);
    return POXPageStatus(rc, code, @"pageStreamingTableBeginV2", error);
}

- (BOOL)streamingTableSetBatchSize:(NSInteger)batchSize error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStreamingTableSetBatchSize");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_streaming_table_set_batch_size(
                             _handle, (uintptr_t)batchSize, &code),
                         code, @"pageStreamingTableSetBatchSize", error);
}

- (NSInteger)streamingTablePendingRowCount {
    if (!_handle)
        return 0;
    return (NSInteger)pdf_page_builder_streaming_table_pending_row_count(_handle);
}

- (NSInteger)streamingTableBatchCount {
    if (!_handle)
        return 0;
    return (NSInteger)pdf_page_builder_streaming_table_batch_count(_handle);
}

- (BOOL)streamingTableFlush:(NSError**)error {
    POX_PAGE_GUARD(@"pageStreamingTableFlush");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_streaming_table_flush(_handle, &code), code,
                         @"pageStreamingTableFlush", error);
}

- (BOOL)streamingTablePushRow:(NSArray<NSString*>*)cells error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStreamingTablePushRow");
    NSUInteger n = cells.count;
    const char** c = n ? (const char**)malloc(sizeof(char*) * n) : NULL;
    for (NSUInteger i = 0; i < n; ++i)
        c[i] = cells[i].UTF8String;
    int32_t code = 0;
    int32_t rc =
        pdf_page_builder_streaming_table_push_row(_handle, (uintptr_t)n, c, &code);
    if (c)
        free(c);
    return POXPageStatus(rc, code, @"pageStreamingTablePushRow", error);
}

- (BOOL)streamingTablePushRowV2:(NSArray<NSString*>*)cells
                       rowspans:(NSArray<NSNumber*>*)rowspans
                          error:(NSError**)error {
    POX_PAGE_GUARD(@"pageStreamingTablePushRowV2");
    NSUInteger n = cells.count;
    const char** c = n ? (const char**)malloc(sizeof(char*) * n) : NULL;
    for (NSUInteger i = 0; i < n; ++i)
        c[i] = cells[i].UTF8String;
    uintptr_t* spans =
        (rowspans && n) ? (uintptr_t*)malloc(sizeof(uintptr_t) * n) : NULL;
    if (spans)
        for (NSUInteger i = 0; i < n; ++i)
            spans[i] = i < rowspans.count ? (uintptr_t)rowspans[i].integerValue : 1;
    int32_t code = 0;
    int32_t rc = pdf_page_builder_streaming_table_push_row_v2(_handle, (uintptr_t)n, c,
                                                              spans, &code);
    if (c)
        free(c);
    if (spans)
        free(spans);
    return POXPageStatus(rc, code, @"pageStreamingTablePushRowV2", error);
}

- (BOOL)streamingTableFinish:(NSError**)error {
    POX_PAGE_GUARD(@"pageStreamingTableFinish");
    int32_t code = 0;
    return POXPageStatus(pdf_page_builder_streaming_table_finish(_handle, &code), code,
                         @"pageStreamingTableFinish", error);
}

- (BOOL)done:(NSError**)error {
    POX_PAGE_GUARD(@"pageDone");
    int32_t code = 0;
    int32_t rc = pdf_page_builder_done(_handle, &code);
    // _done consumes the native handle on success; never call _free afterward.
    _handle = NULL;
    return POXPageStatus(rc, code, @"pageDone", error);
}

- (void)close {
    if (_handle) {
        pdf_page_builder_free(_handle);
        _handle = NULL;
    }
}

@end

@implementation POXDocumentBuilder {
    FfiDocumentBuilder* _handle;
}

+ (instancetype)createWithError:(NSError**)error {
    int32_t code = 0;
    FfiDocumentBuilder* h = pdf_document_builder_create(&code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"documentBuilderCreate");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

- (instancetype)initWithHandle:(FfiDocumentBuilder*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (void)dealloc {
    if (_handle)
        pdf_document_builder_free(_handle);
}

- (BOOL)checkOpen:(NSString*)op error:(NSError**)error {
    if (!_handle) {
        if (error)
            *error = POXMakeError(0, op);
        return NO;
    }
    return YES;
}

#define POX_DOC_GUARD(op)                                                              \
    if (![self checkOpen:(op) error:error])                                            \
    return NO

- (BOOL)setTitle:(NSString*)title error:(NSError**)error {
    POX_DOC_GUARD(@"setTitle");
    int32_t code = 0;
    return POXPageStatus(
        pdf_document_builder_set_title(_handle, title.UTF8String, &code), code,
        @"setTitle", error);
}

- (BOOL)setAuthor:(NSString*)author error:(NSError**)error {
    POX_DOC_GUARD(@"setAuthor");
    int32_t code = 0;
    return POXPageStatus(
        pdf_document_builder_set_author(_handle, author.UTF8String, &code), code,
        @"setAuthor", error);
}

- (BOOL)setSubject:(NSString*)subject error:(NSError**)error {
    POX_DOC_GUARD(@"setSubject");
    int32_t code = 0;
    return POXPageStatus(
        pdf_document_builder_set_subject(_handle, subject.UTF8String, &code), code,
        @"setSubject", error);
}

- (BOOL)setKeywords:(NSString*)keywords error:(NSError**)error {
    POX_DOC_GUARD(@"setKeywords");
    int32_t code = 0;
    return POXPageStatus(
        pdf_document_builder_set_keywords(_handle, keywords.UTF8String, &code), code,
        @"setKeywords", error);
}

- (BOOL)setCreator:(NSString*)creator error:(NSError**)error {
    POX_DOC_GUARD(@"setCreator");
    int32_t code = 0;
    return POXPageStatus(
        pdf_document_builder_set_creator(_handle, creator.UTF8String, &code), code,
        @"setCreator", error);
}

- (BOOL)onOpen:(NSString*)script error:(NSError**)error {
    POX_DOC_GUARD(@"documentOnOpen");
    int32_t code = 0;
    return POXPageStatus(
        pdf_document_builder_on_open(_handle, script.UTF8String, &code), code,
        @"documentOnOpen", error);
}

- (BOOL)taggedPdfUa1:(NSError**)error {
    POX_DOC_GUARD(@"taggedPdfUa1");
    int32_t code = 0;
    return POXPageStatus(pdf_document_builder_tagged_pdf_ua1(_handle, &code), code,
                         @"taggedPdfUa1", error);
}

- (BOOL)language:(NSString*)lang error:(NSError**)error {
    POX_DOC_GUARD(@"language");
    int32_t code = 0;
    return POXPageStatus(pdf_document_builder_language(_handle, lang.UTF8String, &code),
                         code, @"language", error);
}

- (BOOL)roleMapCustom:(NSString*)custom
             standard:(NSString*)standard
                error:(NSError**)error {
    POX_DOC_GUARD(@"roleMap");
    int32_t code = 0;
    return POXPageStatus(pdf_document_builder_role_map(_handle, custom.UTF8String,
                                                       standard.UTF8String, &code),
                         code, @"roleMap", error);
}

- (BOOL)registerEmbeddedFont:(NSString*)name
                        font:(POXEmbeddedFont*)font
                       error:(NSError**)error {
    POX_DOC_GUARD(@"registerEmbeddedFont");
    int32_t code = 0;
    // Pass the native handle WITHOUT consuming it first — on error it must
    // remain valid (and owned by the POXEmbeddedFont wrapper).
    EmbeddedFont* fh = [font takeHandle];
    int32_t rc = pdf_document_builder_register_embedded_font(_handle, name.UTF8String,
                                                             fh, &code);
    if (rc != 0 || code != 0) {
        // Not consumed: hand ownership back to the wrapper so it can free it.
        if (fh) {
            POXEmbeddedFont* re = [[POXEmbeddedFont alloc] initWithHandle:fh];
            (void)re; // re's dealloc will free fh
        }
        if (error)
            *error = POXMakeError(code, @"registerEmbeddedFont");
        return NO;
    }
    // Success: the builder owns fh; the wrapper was already nulled by takeHandle.
    return YES;
}

- (POXPageBuilder*)a4PageWithError:(NSError**)error {
    if (![self checkOpen:@"a4Page" error:error])
        return nil;
    int32_t code = 0;
    FfiPageBuilder* h = pdf_document_builder_a4_page(_handle, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"a4Page");
        return nil;
    }
    return [[POXPageBuilder alloc] initWithHandle:h];
}

- (POXPageBuilder*)letterPageWithError:(NSError**)error {
    if (![self checkOpen:@"letterPage" error:error])
        return nil;
    int32_t code = 0;
    FfiPageBuilder* h = pdf_document_builder_letter_page(_handle, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"letterPage");
        return nil;
    }
    return [[POXPageBuilder alloc] initWithHandle:h];
}

- (POXPageBuilder*)pageWithWidth:(float)width
                          height:(float)height
                           error:(NSError**)error {
    if (![self checkOpen:@"page" error:error])
        return nil;
    int32_t code = 0;
    FfiPageBuilder* h = pdf_document_builder_page(_handle, width, height, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"page");
        return nil;
    }
    return [[POXPageBuilder alloc] initWithHandle:h];
}

- (NSData*)buildWithError:(NSError**)error {
    if (!_handle) {
        if (error)
            *error = POXMakeError(0, @"build");
        return nil;
    }
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = pdf_document_builder_build(_handle, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"build", error);
}

- (BOOL)saveToPath:(NSString*)path error:(NSError**)error {
    POX_DOC_GUARD(@"builderSave");
    int32_t code = 0;
    return POXPageStatus(pdf_document_builder_save(_handle, path.UTF8String, &code),
                         code, @"builderSave", error);
}

- (BOOL)saveEncryptedToPath:(NSString*)path
               userPassword:(NSString*)userPassword
              ownerPassword:(NSString*)ownerPassword
                      error:(NSError**)error {
    POX_DOC_GUARD(@"builderSaveEncrypted");
    int32_t code = 0;
    return POXPageStatus(pdf_document_builder_save_encrypted(
                             _handle, path.UTF8String, userPassword.UTF8String,
                             ownerPassword.UTF8String, &code),
                         code, @"builderSaveEncrypted", error);
}

- (NSData*)toBytesEncryptedWithUserPassword:(NSString*)userPassword
                              ownerPassword:(NSString*)ownerPassword
                                      error:(NSError**)error {
    if (!_handle) {
        if (error)
            *error = POXMakeError(0, @"builderToBytesEncrypted");
        return nil;
    }
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = pdf_document_builder_to_bytes_encrypted(
        _handle, userPassword.UTF8String, ownerPassword.UTF8String, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"builderToBytesEncrypted", error);
}

- (void)close {
    if (_handle) {
        pdf_document_builder_free(_handle);
        _handle = NULL;
    }
}

@end

// ── Phase-6: digital signatures / PKI / timestamps / conformance ─────────────

// Copy a const byte buffer (NOT owned by the caller — do NOT free_bytes) into
// NSData. A null pointer is treated as a failure (sets `error`).
static NSData* _Nullable POXCopyConstBytes(const uint8_t* p, NSUInteger len,
                                           int32_t code, NSString* op,
                                           NSError** error) {
    if (p == NULL) {
        if (error)
            *error = POXMakeError(code, op);
        return nil;
    }
    return [NSData dataWithBytes:p length:len];
}

@implementation POXCertificate {
    void* _handle;
}

+ (instancetype)loadFromBytes:(NSData*)data
                     password:(NSString*)password
                        error:(NSError**)error {
    int32_t code = 0;
    void* h = pdf_certificate_load_from_bytes(data.bytes, (int32_t)data.length,
                                              password.UTF8String, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"certificateLoadFromBytes");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

+ (instancetype)loadFromPemCert:(NSString*)certPem
                         keyPem:(NSString*)keyPem
                          error:(NSError**)error {
    int32_t code = 0;
    void* h =
        pdf_certificate_load_from_pem(certPem.UTF8String, keyPem.UTF8String, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"certificateLoadFromPem");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

- (instancetype)initWithHandle:(void*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

// Expose the raw handle to sibling Phase-6 types in this translation unit.
- (void*)POX_handle {
    return _handle;
}

- (NSString*)subjectError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_certificate_get_subject(_handle, &code), code,
                         @"certificateSubject", error);
}
- (NSString*)issuerError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_certificate_get_issuer(_handle, &code), code,
                         @"certificateIssuer", error);
}
- (NSString*)serialError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_certificate_get_serial(_handle, &code), code,
                         @"certificateSerial", error);
}
- (BOOL)validityNotBefore:(int64_t*)notBefore
                 notAfter:(int64_t*)notAfter
                    error:(NSError**)error {
    int32_t code = 0;
    int64_t nb = 0, na = 0;
    pdf_certificate_get_validity(_handle, &nb, &na, &code);
    if (code != 0) {
        if (error)
            *error = POXMakeError(code, @"certificateValidity");
        return NO;
    }
    if (notBefore)
        *notBefore = nb;
    if (notAfter)
        *notAfter = na;
    return YES;
}
- (int32_t)isValidError:(NSError**)error {
    int32_t code = 0;
    int32_t v = pdf_certificate_is_valid(_handle, &code);
    if (v < 0 && error)
        *error = POXMakeError(code, @"certificateIsValid");
    return v;
}

- (void)close {
    if (_handle) {
        pdf_certificate_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_certificate_free(_handle);
}

@end

@implementation POXSignatureInfo {
    FfiSignatureInfo* _handle;
}

- (instancetype)initWithHandle:(FfiSignatureInfo*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (NSString*)signerNameError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_signature_get_signer_name(_handle, &code), code,
                         @"signatureSignerName", error);
}
- (NSString*)signingReasonError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_signature_get_signing_reason(_handle, &code), code,
                         @"signatureSigningReason", error);
}
- (NSString*)signingLocationError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_signature_get_signing_location(_handle, &code), code,
                         @"signatureSigningLocation", error);
}
- (int64_t)signingTimeError:(NSError**)error {
    int32_t code = 0;
    int64_t t = pdf_signature_get_signing_time(_handle, &code);
    if (code != 0 && error)
        *error = POXMakeError(code, @"signatureSigningTime");
    return t;
}
- (POXCertificate*)certificateError:(NSError**)error {
    int32_t code = 0;
    void* h = pdf_signature_get_certificate(_handle, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"signatureCertificate");
        return nil;
    }
    return [[POXCertificate alloc] initWithHandle:h];
}
- (int32_t)padesLevelError:(NSError**)error {
    int32_t code = 0;
    int32_t lvl = pdf_signature_get_pades_level(_handle, &code);
    if (lvl < 0 && error)
        *error = POXMakeError(code, @"signaturePadesLevel");
    return lvl;
}
- (BOOL)hasTimestampError:(NSError**)error {
    int32_t code = 0;
    bool has = pdf_signature_has_timestamp(_handle, &code);
    if (code != 0 && error)
        *error = POXMakeError(code, @"signatureHasTimestamp");
    return has ? YES : NO;
}
- (POXTimestamp*)timestampError:(NSError**)error {
    int32_t code = 0;
    void* h = pdf_signature_get_timestamp(_handle, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"signatureTimestamp");
        return nil;
    }
    return [[POXTimestamp alloc] initWithHandle:h];
}
- (BOOL)addTimestamp:(POXTimestamp*)timestamp error:(NSError**)error {
    int32_t code = 0;
    bool ok = pdf_signature_add_timestamp(_handle, [timestamp POX_handle], &code);
    if (!ok && error)
        *error = POXMakeError(code, @"signatureAddTimestamp");
    return ok ? YES : NO;
}
- (int32_t)verifyError:(NSError**)error {
    int32_t code = 0;
    int32_t r = pdf_signature_verify(_handle, &code);
    if (r < 0 && error)
        *error = POXMakeError(code, @"signatureVerify");
    return r;
}
- (int32_t)verifyDetached:(NSData*)pdf error:(NSError**)error {
    int32_t code = 0;
    int32_t r =
        pdf_signature_verify_detached(_handle, pdf.bytes, (uintptr_t)pdf.length, &code);
    if (r < 0 && error)
        *error = POXMakeError(code, @"signatureVerifyDetached");
    return r;
}

- (void)close {
    if (_handle) {
        pdf_signature_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_signature_free(_handle);
}

@end

@implementation POXTimestamp {
    void* _handle;
}

+ (instancetype)parse:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    void* h = pdf_timestamp_parse(data.bytes, (uintptr_t)data.length, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"timestampParse");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

- (instancetype)initWithHandle:(void*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (void*)POX_handle {
    return _handle;
}

- (NSData*)tokenError:(NSError**)error {
    int32_t code = 0;
    uintptr_t len = 0;
    const uint8_t* p = pdf_timestamp_get_token(_handle, &len, &code);
    return POXCopyConstBytes(p, (NSUInteger)len, code, @"timestampToken", error);
}
- (NSData*)messageImprintError:(NSError**)error {
    int32_t code = 0;
    uintptr_t len = 0;
    const uint8_t* p = pdf_timestamp_get_message_imprint(_handle, &len, &code);
    return POXCopyConstBytes(p, (NSUInteger)len, code, @"timestampMessageImprint",
                             error);
}
- (int64_t)timeError:(NSError**)error {
    int32_t code = 0;
    int64_t t = pdf_timestamp_get_time(_handle, &code);
    if (code != 0 && error)
        *error = POXMakeError(code, @"timestampTime");
    return t;
}
- (NSString*)serialError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_timestamp_get_serial(_handle, &code), code,
                         @"timestampSerial", error);
}
- (NSString*)tsaNameError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_timestamp_get_tsa_name(_handle, &code), code,
                         @"timestampTsaName", error);
}
- (NSString*)policyOidError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_timestamp_get_policy_oid(_handle, &code), code,
                         @"timestampPolicyOid", error);
}
- (int32_t)hashAlgorithmError:(NSError**)error {
    int32_t code = 0;
    int32_t a = pdf_timestamp_get_hash_algorithm(_handle, &code);
    if (a < 0 && error)
        *error = POXMakeError(code, @"timestampHashAlgorithm");
    return a;
}
- (BOOL)verifyError:(NSError**)error {
    int32_t code = 0;
    bool ok = pdf_timestamp_verify(_handle, &code);
    if (!ok && code != 0 && error)
        *error = POXMakeError(code, @"timestampVerify");
    return ok ? YES : NO;
}

- (void)close {
    if (_handle) {
        pdf_timestamp_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_timestamp_free(_handle);
}

@end

@implementation POXTsaClient {
    void* _handle;
}

+ (instancetype)createWithUrl:(NSString*)url
                     username:(NSString*)username
                     password:(NSString*)password
                      timeout:(int32_t)timeout
                     hashAlgo:(int32_t)hashAlgo
                     useNonce:(BOOL)useNonce
                      certReq:(BOOL)certReq
                        error:(NSError**)error {
    int32_t code = 0;
    void* h =
        pdf_tsa_client_create(url.UTF8String, username ? username.UTF8String : NULL,
                              password ? password.UTF8String : NULL, timeout, hashAlgo,
                              useNonce ? true : false, certReq ? true : false, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"tsaClientCreate");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

- (instancetype)initWithHandle:(void*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (POXTimestamp*)requestTimestamp:(NSData*)data error:(NSError**)error {
    int32_t code = 0;
    void* h =
        pdf_tsa_request_timestamp(_handle, data.bytes, (uintptr_t)data.length, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"tsaRequestTimestamp");
        return nil;
    }
    return [[POXTimestamp alloc] initWithHandle:h];
}
- (POXTimestamp*)requestTimestampHash:(NSData*)hash
                             hashAlgo:(int32_t)hashAlgo
                                error:(NSError**)error {
    int32_t code = 0;
    void* h = pdf_tsa_request_timestamp_hash(_handle, hash.bytes,
                                             (uintptr_t)hash.length, hashAlgo, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"tsaRequestTimestampHash");
        return nil;
    }
    return [[POXTimestamp alloc] initWithHandle:h];
}

- (void)close {
    if (_handle) {
        pdf_tsa_client_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_tsa_client_free(_handle);
}

@end

@implementation POXDss {
    void* _handle;
}

- (instancetype)initWithHandle:(void*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (int32_t)certCount {
    return pdf_dss_cert_count(_handle);
}
- (int32_t)crlCount {
    return pdf_dss_crl_count(_handle);
}
- (int32_t)ocspCount {
    return pdf_dss_ocsp_count(_handle);
}
- (int32_t)vriCount {
    return pdf_dss_vri_count(_handle);
}
- (NSData*)certAtIndex:(int32_t)index error:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = pdf_dss_get_cert(_handle, index, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"dssGetCert", error);
}
- (NSData*)crlAtIndex:(int32_t)index error:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = pdf_dss_get_crl(_handle, index, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"dssGetCrl", error);
}
- (NSData*)ocspAtIndex:(int32_t)index error:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = pdf_dss_get_ocsp(_handle, index, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"dssGetOcsp", error);
}

- (void)close {
    if (_handle) {
        pdf_dss_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_dss_free(_handle);
}

@end

@implementation POXPdfAResults {
    FfiPdfAResults* _handle;
}
- (instancetype)initWithHandle:(FfiPdfAResults*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}
- (BOOL)isCompliantError:(NSError**)error {
    int32_t code = 0;
    bool ok = pdf_pdf_a_is_compliant(_handle, &code);
    if (code != 0 && error)
        *error = POXMakeError(code, @"pdfAIsCompliant");
    return ok ? YES : NO;
}
- (int32_t)errorCount {
    return pdf_pdf_a_error_count(_handle);
}
- (int32_t)warningCount {
    return pdf_pdf_a_warning_count(_handle);
}
- (NSArray<NSString*>*)errors {
    int32_t n = pdf_pdf_a_error_count(_handle);
    NSMutableArray<NSString*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* s = POXTakeString(pdf_pdf_a_get_error(_handle, i, &c), c,
                                    @"pdfAGetError", NULL);
        [out addObject:(s ?: @"")];
    }
    return out;
}
- (void)close {
    if (_handle) {
        pdf_pdf_a_results_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_pdf_a_results_free(_handle);
}
@end

@implementation POXUaResults {
    FfiUaResults* _handle;
}
- (instancetype)initWithHandle:(FfiUaResults*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}
- (BOOL)isAccessibleError:(NSError**)error {
    int32_t code = 0;
    bool ok = pdf_pdf_ua_is_accessible(_handle, &code);
    if (code != 0 && error)
        *error = POXMakeError(code, @"pdfUaIsAccessible");
    return ok ? YES : NO;
}
- (int32_t)errorCount {
    return pdf_pdf_ua_error_count(_handle);
}
- (int32_t)warningCount {
    return pdf_pdf_ua_warning_count(_handle);
}
- (NSArray<NSString*>*)errors {
    int32_t n = pdf_pdf_ua_error_count(_handle);
    NSMutableArray<NSString*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* s = POXTakeString(pdf_pdf_ua_get_error(_handle, i, &c), c,
                                    @"pdfUaGetError", NULL);
        [out addObject:(s ?: @"")];
    }
    return out;
}
- (NSArray<NSString*>*)warnings {
    int32_t n = pdf_pdf_ua_warning_count(_handle);
    NSMutableArray<NSString*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* s = POXTakeString(pdf_pdf_ua_get_warning(_handle, i, &c), c,
                                    @"pdfUaGetWarning", NULL);
        [out addObject:(s ?: @"")];
    }
    return out;
}
- (BOOL)stats:(POXUaStats*)stats error:(NSError**)error {
    int32_t code = 0;
    int32_t st = 0, im = 0, tb = 0, fm = 0, an = 0, pg = 0;
    bool ok = pdf_pdf_ua_get_stats(_handle, &st, &im, &tb, &fm, &an, &pg, &code);
    if (!ok) {
        if (error)
            *error = POXMakeError(code, @"pdfUaGetStats");
        return NO;
    }
    if (stats) {
        stats->structElements = st;
        stats->images = im;
        stats->tables = tb;
        stats->forms = fm;
        stats->annotations = an;
        stats->pages = pg;
    }
    return YES;
}
- (void)close {
    if (_handle) {
        pdf_pdf_ua_results_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_pdf_ua_results_free(_handle);
}
@end

@implementation POXPdfXResults {
    FfiPdfXResults* _handle;
}
- (instancetype)initWithHandle:(FfiPdfXResults*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}
- (BOOL)isCompliantError:(NSError**)error {
    int32_t code = 0;
    bool ok = pdf_pdf_x_is_compliant(_handle, &code);
    if (code != 0 && error)
        *error = POXMakeError(code, @"pdfXIsCompliant");
    return ok ? YES : NO;
}
- (int32_t)errorCount {
    return pdf_pdf_x_error_count(_handle);
}
- (NSArray<NSString*>*)errors {
    int32_t n = pdf_pdf_x_error_count(_handle);
    NSMutableArray<NSString*>* out = [NSMutableArray arrayWithCapacity:(n < 0 ? 0 : n)];
    for (int32_t i = 0; i < n; ++i) {
        int32_t c = 0;
        NSString* s = POXTakeString(pdf_pdf_x_get_error(_handle, i, &c), c,
                                    @"pdfXGetError", NULL);
        [out addObject:(s ?: @"")];
    }
    return out;
}
- (void)close {
    if (_handle) {
        pdf_pdf_x_results_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_pdf_x_results_free(_handle);
}
@end

@implementation POXPadesSignOptions
- (instancetype)init {
    if ((self = [super init])) {
        _certs = @[];
        _crls = @[];
        _ocsps = @[];
    }
    return self;
}
@end

// Build three parallel C arrays of (pointer, length) for a list of NSData blobs.
// The returned pointers borrow each NSData's bytes — valid only while `blobs`
// is retained for the duration of the call. Frees nothing; caller frees `ptrs`
// and `lens` via free().
static void POXBuildByteArrays(NSArray<NSData*>* blobs, const uint8_t*** ptrs,
                               uintptr_t** lens, uintptr_t* count) {
    NSUInteger n = blobs.count;
    *count = (uintptr_t)n;
    *ptrs = n ? (const uint8_t**)malloc(sizeof(uint8_t*) * n) : NULL;
    *lens = n ? (uintptr_t*)malloc(sizeof(uintptr_t) * n) : NULL;
    for (NSUInteger i = 0; i < n; ++i) {
        (*ptrs)[i] = (const uint8_t*)blobs[i].bytes;
        (*lens)[i] = (uintptr_t)blobs[i].length;
    }
}

@implementation POXSigning

+ (NSData*)signBytes:(NSData*)pdf
         certificate:(POXCertificate*)certificate
              reason:(NSString*)reason
            location:(NSString*)location
               error:(NSError**)error {
    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p =
        pdf_sign_bytes(pdf.bytes, (uintptr_t)pdf.length, [certificate POX_handle],
                       reason ? reason.UTF8String : NULL,
                       location ? location.UTF8String : NULL, &len, &code);
    return POXTakeBytes(p, (NSUInteger)len, code, @"signBytes", error);
}

+ (NSData*)signBytesPades:(NSData*)pdf
              certificate:(POXCertificate*)certificate
                    level:(int32_t)level
                   tsaUrl:(NSString*)tsaUrl
                   reason:(NSString*)reason
                 location:(NSString*)location
                    certs:(NSArray<NSData*>*)certs
                     crls:(NSArray<NSData*>*)crls
                    ocsps:(NSArray<NSData*>*)ocsps
                    error:(NSError**)error {
    const uint8_t** certPtrs = NULL;
    uintptr_t* certLens = NULL;
    uintptr_t nCerts = 0;
    const uint8_t** crlPtrs = NULL;
    uintptr_t* crlLens = NULL;
    uintptr_t nCrls = 0;
    const uint8_t** ocspPtrs = NULL;
    uintptr_t* ocspLens = NULL;
    uintptr_t nOcsps = 0;
    POXBuildByteArrays(certs ?: @[], &certPtrs, &certLens, &nCerts);
    POXBuildByteArrays(crls ?: @[], &crlPtrs, &crlLens, &nCrls);
    POXBuildByteArrays(ocsps ?: @[], &ocspPtrs, &ocspLens, &nOcsps);

    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p = pdf_sign_bytes_pades(
        pdf.bytes, (uintptr_t)pdf.length, [certificate POX_handle], level,
        tsaUrl ? tsaUrl.UTF8String : NULL, reason ? reason.UTF8String : NULL,
        location ? location.UTF8String : NULL, certPtrs, certLens, nCerts, crlPtrs,
        crlLens, nCrls, ocspPtrs, ocspLens, nOcsps, &len, &code);

    free(certPtrs);
    free(certLens);
    free(crlPtrs);
    free(crlLens);
    free(ocspPtrs);
    free(ocspLens);
    return POXTakeBytes(p, (NSUInteger)len, code, @"signBytesPades", error);
}

+ (NSData*)signBytesPadesOpts:(NSData*)pdf
                      options:(POXPadesSignOptions*)options
                        error:(NSError**)error {
    const uint8_t** certPtrs = NULL;
    uintptr_t* certLens = NULL;
    uintptr_t nCerts = 0;
    const uint8_t** crlPtrs = NULL;
    uintptr_t* crlLens = NULL;
    uintptr_t nCrls = 0;
    const uint8_t** ocspPtrs = NULL;
    uintptr_t* ocspLens = NULL;
    uintptr_t nOcsps = 0;
    POXBuildByteArrays(options.certs ?: @[], &certPtrs, &certLens, &nCerts);
    POXBuildByteArrays(options.crls ?: @[], &crlPtrs, &crlLens, &nCrls);
    POXBuildByteArrays(options.ocsps ?: @[], &ocspPtrs, &ocspLens, &nOcsps);

    PadesSignOptionsC opts;
    memset(&opts, 0, sizeof(opts));
    opts.certificate_handle = [options.certificate POX_handle];
    opts.certs = certPtrs;
    opts.cert_lens = certLens;
    opts.n_certs = nCerts;
    opts.crls = crlPtrs;
    opts.crl_lens = crlLens;
    opts.n_crls = nCrls;
    opts.ocsps = ocspPtrs;
    opts.ocsp_lens = ocspLens;
    opts.n_ocsps = nOcsps;
    opts.tsa_url = options.tsaUrl ? options.tsaUrl.UTF8String : NULL;
    opts.reason = options.reason ? options.reason.UTF8String : NULL;
    opts.location = options.location ? options.location.UTF8String : NULL;
    opts.level = options.level;

    uintptr_t len = 0;
    int32_t code = 0;
    uint8_t* p =
        pdf_sign_bytes_pades_opts(pdf.bytes, (uintptr_t)pdf.length, &opts, &len, &code);

    free(certPtrs);
    free(certLens);
    free(crlPtrs);
    free(crlLens);
    free(ocspPtrs);
    free(ocspLens);
    return POXTakeBytes(p, (NSUInteger)len, code, @"signBytesPadesOpts", error);
}

+ (void)setLogLevel:(int32_t)level {
    pdf_oxide_set_log_level(level);
}
+ (int32_t)logLevel {
    return pdf_oxide_get_log_level();
}

@end

// ── Phase-7: barcodes / OCR / element list / merge + timestamp ───────────────

@implementation POXBarcode {
    FfiBarcodeImage* _handle;
}

+ (instancetype)generateQrCode:(NSString*)data
               errorCorrection:(int32_t)errorCorrection
                        sizePx:(int32_t)sizePx
                         error:(NSError**)error {
    int32_t code = 0;
    FfiBarcodeImage* h =
        pdf_generate_qr_code(data.UTF8String, errorCorrection, sizePx, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"generateQrCode");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

+ (instancetype)generateBarcode:(NSString*)data
                         format:(int32_t)format
                         sizePx:(int32_t)sizePx
                          error:(NSError**)error {
    int32_t code = 0;
    FfiBarcodeImage* h = pdf_generate_barcode(data.UTF8String, format, sizePx, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"generateBarcode");
        return nil;
    }
    return [[self alloc] initWithHandle:h];
}

- (instancetype)initWithHandle:(FfiBarcodeImage*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (const FfiBarcodeImage*)POX_handle {
    return _handle;
}

- (NSString*)dataError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_barcode_get_data(_handle, &code), code, @"barcodeData",
                         error);
}

- (int32_t)formatError:(NSError**)error {
    int32_t code = 0;
    int32_t f = pdf_barcode_get_format(_handle, &code);
    if (f < 0 && error)
        *error = POXMakeError(code, @"barcodeFormat");
    return f;
}

- (float)confidenceError:(NSError**)error {
    int32_t code = 0;
    float c = pdf_barcode_get_confidence(_handle, &code);
    if (code != 0 && error)
        *error = POXMakeError(code, @"barcodeConfidence");
    return c;
}

- (NSData*)imagePngWithSizePx:(int32_t)sizePx error:(NSError**)error {
    int32_t outLen = 0, code = 0;
    uint8_t* p = pdf_barcode_get_image_png(_handle, sizePx, &outLen, &code);
    return POXTakeBytes(p, (outLen < 0 ? 0 : (NSUInteger)outLen), code,
                        @"barcodeImagePng", error);
}

- (NSString*)svgWithSizePx:(int32_t)sizePx error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_barcode_get_svg(_handle, sizePx, &code), code,
                         @"barcodeSvg", error);
}

- (void)close {
    if (_handle) {
        pdf_barcode_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_barcode_free(_handle);
}

@end

@implementation POXOcrEngine {
    void* _handle;
}

+ (instancetype)createWithDetModelPath:(NSString*)detModelPath
                          recModelPath:(NSString*)recModelPath
                              dictPath:(NSString*)dictPath
                                 error:(NSError**)error {
    int32_t code = 0;
    void* h = pdf_ocr_engine_create(detModelPath.UTF8String, recModelPath.UTF8String,
                                    dictPath.UTF8String, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"ocrEngineCreate");
        return nil;
    }
    POXOcrEngine* e = [[self alloc] init];
    if (e)
        e->_handle = h;
    return e;
}

- (void*)POX_engineHandle {
    return _handle;
}

- (void)close {
    if (_handle) {
        pdf_ocr_engine_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_ocr_engine_free(_handle);
}

@end

@implementation POXElementList {
    FfiElementList* _handle;
}

- (instancetype)initWithHandle:(FfiElementList*)handle {
    if ((self = [super init])) {
        _handle = handle;
    }
    return self;
}

- (int32_t)count {
    if (!_handle)
        return 0;
    return pdf_oxide_element_count(_handle);
}

- (NSString*)typeAtIndex:(int32_t)index error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_oxide_element_get_type(_handle, index, &code), code,
                         @"elementType", error);
}

- (NSString*)textAtIndex:(int32_t)index error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_oxide_element_get_text(_handle, index, &code), code,
                         @"elementText", error);
}

- (POXBbox)rectAtIndex:(int32_t)index error:(NSError**)error {
    int32_t code = 0;
    float x = 0, y = 0, w = 0, h = 0;
    pdf_oxide_element_get_rect(_handle, index, &x, &y, &w, &h, &code);
    if (code != 0 && error)
        *error = POXMakeError(code, @"elementRect");
    POXBbox box = {x, y, w, h};
    return box;
}

- (NSString*)toJsonWithError:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_oxide_elements_to_json(_handle, &code), code,
                         @"elementsToJson", error);
}

- (void)close {
    if (_handle) {
        pdf_oxide_elements_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_oxide_elements_free(_handle);
}

@end

@implementation POXTools

+ (NSData*)merge:(NSArray<NSString*>*)paths error:(NSError**)error {
    NSUInteger count = paths.count;
    const char** cpaths = count ? (const char**)malloc(sizeof(char*) * count) : NULL;
    for (NSUInteger i = 0; i < count; ++i)
        cpaths[i] = paths[i].UTF8String;
    int32_t dataLen = 0, code = 0;
    uint8_t* p = pdf_merge(cpaths, (int32_t)count, &dataLen, &code);
    if (cpaths)
        free(cpaths);
    return POXTakeBytes(p, (dataLen < 0 ? 0 : (NSUInteger)dataLen), code, @"merge",
                        error);
}

+ (NSData*)addTimestamp:(NSData*)pdfData
               sigIndex:(int32_t)sigIndex
                 tsaUrl:(NSString*)tsaUrl
                  error:(NSError**)error {
    uint8_t* out = NULL;
    uintptr_t outLen = 0;
    int32_t code = 0;
    bool ok = pdf_add_timestamp(pdfData.bytes, (uintptr_t)pdfData.length, sigIndex,
                                tsaUrl.UTF8String, &out, &outLen, &code);
    if (!ok || out == NULL) {
        if (out)
            free_bytes(out);
        if (error)
            *error = POXMakeError(code, @"addTimestamp");
        return nil;
    }
    return POXTakeBytes(out, (NSUInteger)outLen, code, @"addTimestamp", error);
}

@end

// ── Crypto / FIPS / models / config / renderer ───────────────────────────────

@implementation POXCrypto

+ (NSString*)activeProviderWithError:(NSError**)error {
    char* s = pdf_oxide_crypto_active_provider();
    return POXTakeString(s, -1, @"cryptoActiveProvider", error);
}
+ (NSString*)cbomWithError:(NSError**)error {
    char* s = pdf_oxide_crypto_cbom();
    return POXTakeString(s, -1, @"cryptoCbom", error);
}
+ (NSString*)inventoryWithError:(NSError**)error {
    char* s = pdf_oxide_crypto_inventory();
    return POXTakeString(s, -1, @"cryptoInventory", error);
}
+ (NSString*)policyWithError:(NSError**)error {
    char* s = pdf_oxide_crypto_policy();
    return POXTakeString(s, -1, @"cryptoPolicy", error);
}
+ (int32_t)fipsAvailable {
    return pdf_oxide_crypto_fips_available();
}
+ (int32_t)useFips {
    return pdf_oxide_crypto_use_fips();
}
+ (int32_t)setPolicy:(NSString*)spec {
    return pdf_oxide_crypto_set_policy(spec.UTF8String);
}

@end

@implementation POXModels

+ (NSString*)manifestWithError:(NSError**)error {
    char* s = pdf_oxide_model_manifest();
    return POXTakeString(s, -1, @"modelManifest", error);
}
+ (int32_t)prefetchAvailable {
    return pdf_oxide_prefetch_available();
}
+ (NSString*)prefetchModels:(NSString*)languagesCsv error:(NSError**)error {
    int32_t code = 0;
    return POXTakeString(pdf_oxide_prefetch_models(languagesCsv.UTF8String, &code),
                         code, @"prefetchModels", error);
}

@end

@implementation POXConfig

+ (int64_t)setMaxOpsPerStream:(int64_t)limit {
    return pdf_oxide_set_max_ops_per_stream(limit);
}
+ (int32_t)setPreserveUnmappedGlyphs:(int32_t)preserve {
    return pdf_oxide_set_preserve_unmapped_glyphs(preserve);
}

@end

@implementation POXRenderer {
    void* _handle;
}

+ (instancetype)createWithDpi:(int32_t)dpi
                       format:(int32_t)format
                      quality:(int32_t)quality
                    antiAlias:(BOOL)antiAlias
                        error:(NSError**)error {
    int32_t code = 0;
    void* h =
        pdf_create_renderer(dpi, format, quality, antiAlias ? true : false, &code);
    if (!h) {
        if (error)
            *error = POXMakeError(code, @"createRenderer");
        return nil;
    }
    POXRenderer* r = [[self alloc] init];
    if (r)
        r->_handle = h;
    return r;
}

- (void)close {
    if (_handle) {
        pdf_renderer_free(_handle);
        _handle = NULL;
    }
}
- (void)dealloc {
    if (_handle)
        pdf_renderer_free(_handle);
}

@end
