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
        CHECK([[doc toMarkdownAllWithError:&err] length] > 0); // toMarkdownAll
        CHECK([[doc extractStructuredJson:0
                                    error:&err] length] > 0); // extractStructuredJson

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
