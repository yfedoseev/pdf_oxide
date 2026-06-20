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
        [out addObject:[[POXFont alloc]
                           initWithName:(name ?: @"")
                                   type:(type ?: @"")encoding:(encoding ?: @"")embedded
                                       :(embedded ? YES : NO)subset
                                       :(subset ? YES : NO)]];
    }
    pdf_oxide_font_list_free(list);
    return out;
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
        [out addObject:[[POXAnnotation alloc]
                           initWithType:(type ?: @"")
                                subtype:(subtype ?: @"")content:(content ?: @"")author
                                       :(author ?: @"")rect:rect
                            borderWidth:borderWidth]];
    }
    pdf_oxide_annotation_list_free(list);
    return out;
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
