// basic_extraction — build a PDF from Markdown, then extract it back.
// Run in CI as a smoke example (no external fixture).
#import "POXPdfOxide.h"
#import <Foundation/Foundation.h>

int main(void) {
    @autoreleasepool {
        NSError* err = nil;
        POXPdf* pdf = [POXPdf fromMarkdown:@"# Hello pdf_oxide\n\nThis is an "
                                           @"**Objective-C** binding smoke example.\n"
                                     error:&err];
        if (!pdf) {
            fprintf(stderr, "error: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }

        POXDocument* doc = [POXDocument openFromBytes:[pdf toBytesWithError:&err]
                                                error:&err];
        if (!doc) {
            fprintf(stderr, "error: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }

        printf("pages:   %ld\n", (long)[doc pageCountError:&err]);
        POXVersion ver = [doc version];
        printf("version: %d.%d\n", ver.major, ver.minor);
        printf("--- text (page 0) ---\n%s\n",
               [doc extractText:0 error:&err].UTF8String);
        printf("--- markdown (all) ---\n%s\n",
               [doc toMarkdownAllWithError:&err].UTF8String);
        return 0;
    }
}
