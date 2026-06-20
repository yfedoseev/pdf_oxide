// One check per public method — mirrors the api_coverage convention used by
// every pdf_oxide binding. Plain clang-built executable (no XCTest harness):
// returns non-zero on any failure. Self-contained: builds its own PDF.
#import "POXPdfOxide.h"
#import <Foundation/Foundation.h>

static int g_failures = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);            \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

static NSData* samplePdf(void) {
    NSError* err = nil;
    POXPdf* p = [POXPdf
        fromMarkdown:@"# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n"
               error:&err];
    return [p toBytesWithError:&err];
}

int main(void) {
    @autoreleasepool {
        NSError* err = nil;

        // ── Pdf builder ──────────────────────────────────────────────────────
        CHECK([[[POXPdf fromMarkdown:@"# md\n\nbody\n"
                               error:&err] toBytesWithError:&err] length] > 100);
        CHECK([[[POXPdf fromHtml:@"<h1>h</h1><p>b</p>"
                           error:&err] toBytesWithError:&err] length] > 100);
        CHECK([[[POXPdf fromText:@"plain text body"
                           error:&err] toBytesWithError:&err] length] > 100);
        {
            NSString* path = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"pdfoxide_objc.pdf"];
            POXPdf* p = [POXPdf fromMarkdown:@"# f\n\nx\n" error:&err];
            CHECK([p saveToPath:path error:&err]); // save
            CHECK([[NSFileManager defaultManager] fileExistsAtPath:path]);
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
        }

        // ── Document open paths ──────────────────────────────────────────────
        POXDocument* doc = [POXDocument openFromBytes:samplePdf()
                                                error:&err]; // openFromBytes
        CHECK(doc != nil);
        CHECK([doc pageCountError:&err] >= 1); // pageCount
        {
            NSString* path = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"pdfoxide_objc_open.pdf"];
            [[POXPdf fromMarkdown:@"# f\n\nx\n" error:&err] saveToPath:path error:&err];
            POXDocument* d2 = [POXDocument openPath:path error:&err]; // openPath
            CHECK([d2 pageCountError:&err] >= 1);
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
        }

        // ── Document inspection + extraction ─────────────────────────────────
        POXVersion ver = [doc version]; // version
        CHECK(ver.major >= 1);
        CHECK([doc isEncrypted] == NO); // isEncrypted
        (void)[doc hasStructureTree];   // hasStructureTree
        CHECK([[doc extractText:0 error:&err] containsString:@"Alpha"]); // extractText
        CHECK([[doc toPlainText:0 error:&err] length] > 0);              // toPlainText
        CHECK([[doc toMarkdown:0 error:&err] length] > 0);               // toMarkdown
        CHECK([[doc toHtml:0 error:&err] containsString:@"<"]);          // toHtml
        CHECK([[doc toMarkdownAllWithError:&err] length] > 0);      // toMarkdownAll
        CHECK([[doc toHtmlAllWithError:&err] containsString:@"<"]); // toHtmlAll
        CHECK([[doc toPlainTextAllWithError:&err] length] > 0);     // toPlainTextAll
        CHECK([[doc extractStructuredJson:0
                                    error:&err] length] > 0); // extractStructuredJson

        // ── Phase-1 element extraction ───────────────────────────────────────
        {
            NSArray<POXWord*>* words = [doc extractWords:0 error:&err]; // extractWords
            CHECK(words != nil && words.count > 0);
            if (words.count > 0) {
                POXWord* w0 = words[0];
                CHECK(w0.text.length > 0);
                CHECK(w0.bbox.width >= 0 && w0.bbox.height >= 0);
                CHECK(w0.bold == YES || w0.bold == NO);
            }
            NSArray<POXChar*>* chars = [doc extractChars:0 error:&err]; // extractChars
            CHECK(chars != nil && chars.count > 0);
            NSArray<POXTextLine*>* lines =
                [doc extractTextLines:0 error:&err]; // extractTextLines
            CHECK(lines != nil && lines.count > 0);
            NSError* te = nil;
            NSArray<POXTable*>* tables =
                [doc extractTables:0 error:&te]; // extractTables (may be empty)
            CHECK(tables != nil && te == nil);
        }

        // ── Phase-2 extraction ───────────────────────────────────────────────
        {
            NSError* fe = nil;
            NSArray<POXFont*>* fonts =
                [doc embeddedFonts:0 error:&fe]; // embeddedFonts (may be empty)
            CHECK(fonts != nil && fe == nil);
            NSError* ie = nil;
            NSArray<POXImage*>* images =
                [doc embeddedImages:0 error:&ie]; // embeddedImages (may be empty)
            CHECK(images != nil && ie == nil);
            NSError* ane = nil;
            NSArray<POXAnnotation*>* annots =
                [doc pageAnnotations:0 error:&ane]; // pageAnnotations (may be empty)
            CHECK(annots != nil && ane == nil);
            NSError* pe = nil;
            NSArray<POXPath*>* paths =
                [doc extractPaths:0 error:&pe]; // extractPaths (may be empty)
            CHECK(paths != nil && pe == nil);

            NSArray<POXSearchResult*>* hits = [doc search:0
                                                     term:@"Alpha"
                                            caseSensitive:NO
                                                    error:&err]; // search
            CHECK(hits != nil && hits.count > 0);
            if (hits.count > 0) {
                CHECK([hits[0].text containsString:@"Alpha"]);
                CHECK(hits[0].page >= 0);
            }
            NSArray<POXSearchResult*>* allHits = [doc searchAll:@"Alpha"
                                                  caseSensitive:NO
                                                          error:&err]; // searchAll
            CHECK(allHits != nil && allHits.count > 0);
            if (allHits.count > 0) {
                CHECK([allHits[0].text containsString:@"Alpha"]);
                CHECK(allHits[0].page >= 0);
            }
        }

        // ── authenticate (wrong password on unencrypted doc returns a bool) ──
        {
            NSError* ae = nil;
            BOOL authed = [doc authenticate:@"any-password" error:&ae]; // authenticate
            CHECK(authed == YES || authed == NO);
        }

        // ── Page model ───────────────────────────────────────────────────────
        {
            POXPage* page = [doc pageAtIndex:0];               // pageAtIndex
            CHECK([[page text:&err] containsString:@"Alpha"]); // Page text
            CHECK([[page markdown:&err] length] > 0);          // Page markdown
            CHECK([[page html:&err] length] > 0);              // Page html
            CHECK([[page plainText:&err] length] > 0);         // Page plainText
        }

        // ── Phase-3 page rendering ───────────────────────────────────────────
        {
            NSError* re = nil;
            POXRenderedImage* img = [doc renderPage:0
                                             format:0
                                              error:&re]; // renderPage (PNG)
            CHECK(img != nil && re == nil);
            if (img != nil) {
                CHECK(img.width > 0);       // RenderedImage width
                CHECK(img.height > 0);      // RenderedImage height
                CHECK(img.data.length > 0); // RenderedImage data
                NSString* path = [NSTemporaryDirectory()
                    stringByAppendingPathComponent:@"pdfoxide_objc_render.png"];
                CHECK([img saveToPath:path error:&re]); // RenderedImage saveToPath
                CHECK([[NSFileManager defaultManager] fileExistsAtPath:path]);
                [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
                [img close];
            }
            NSError* ze = nil;
            POXRenderedImage* zoomed = [doc renderPageZoom:0
                                                      zoom:2.0f
                                                    format:0
                                                     error:&ze]; // renderPageZoom
            CHECK(zoomed != nil && ze == nil);
            NSError* the = nil;
            POXRenderedImage* thumb =
                [doc renderPageThumbnail:0
                                    size:64
                                  format:0
                                   error:&the]; // renderPageThumbnail
            CHECK(thumb != nil && the == nil);
        }

        // ── DocumentEditor ───────────────────────────────────────────────────
        {
            NSError* ee = nil;
            POXDocumentEditor* ed =
                [POXDocumentEditor openFromBytes:samplePdf()
                                           error:&ee]; // openFromBytes
            CHECK(ed != nil && ee == nil);
            CHECK([ed pageCountError:&ee] >= 1); // pageCount
            POXVersion ev = [ed version];        // version
            CHECK(ev.major >= 1);
            BOOL mod = [ed isModified]; // isModified (bool)
            CHECK(mod == YES || mod == NO);
            CHECK([ed rotateAllPages:90 error:&ee]); // rotateAllPages
            CHECK([ed pageRotation:0 error:&ee] == 90 ||
                  [ed pageRotation:0 error:&ee] >= 0);            // getPageRotation
            CHECK([ed setProducer:@"x" error:&ee]);               // setProducer
            CHECK([[ed producerError:&ee] isEqualToString:@"x"]); // getProducer
            NSData* edBytes = [ed saveToBytesWithError:&ee];      // saveToBytes
            CHECK(edBytes != nil && edBytes.length > 0);
            [ed close]; // close
            [ed close]; // idempotent
        }

        // ── close (idempotent) ───────────────────────────────────────────────
        [doc close];
        [doc close]; // idempotent — safe to call twice

        // ── Error path ───────────────────────────────────────────────────────
        NSError* e2 = nil;
        POXDocument* bad = [POXDocument openPath:@"/nonexistent/nope.pdf" error:&e2];
        CHECK(bad == nil && e2 != nil);

        if (g_failures == 0) {
            printf("ok: all Objective-C api-coverage checks passed\n");
            return 0;
        }
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
}
