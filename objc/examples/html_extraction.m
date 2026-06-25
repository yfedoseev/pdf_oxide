// html_extraction — build a PDF from Markdown, then render it back to HTML.
// Shared-scenario regression example. Returns non-zero on assertion failure.
#import "POXPdfOxide.h"
#import <Foundation/Foundation.h>

int main(void) {
    @autoreleasepool {
        NSError* err = nil;
        POXPdf* pdf = [POXPdf fromMarkdown:@"# Hello pdf_oxide\n\nThis is a "
                                           @"**Objective-C** regression example.\n"
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

        NSString* html = [doc toHtmlAllWithError:&err];
        if (!html) {
            fprintf(stderr, "error: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        printf("%s\n", html.UTF8String);

        if (![html containsString:@"<"]) {
            fprintf(stderr, "assertion failed: html missing '<'\n");
            return 1;
        }
        if (![html containsString:@"pdf_oxide"]) {
            fprintf(stderr, "assertion failed: html missing 'pdf_oxide'\n");
            return 1;
        }
        printf("HTML OK\n");
        return 0;
    }
}
