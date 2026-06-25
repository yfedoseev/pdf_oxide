// words_geometry — build a PDF from Markdown, then extract positioned words.
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

        NSArray<POXWord*>* words = [doc extractWords:0 error:&err];
        if (!words) {
            fprintf(stderr, "error: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        if (words.count == 0) {
            fprintf(stderr, "assertion failed: no words extracted\n");
            return 1;
        }

        POXWord* first = words[0];
        printf("words: %lu\n", (unsigned long)words.count);
        printf("first word: \"%s\"\n", first.text.UTF8String);
        printf("bbox: x=%f y=%f w=%f h=%f\n", first.bbox.x, first.bbox.y,
               first.bbox.width, first.bbox.height);

        if (![first.text isEqualToString:@"Hello"]) {
            fprintf(stderr, "assertion failed: first word != \"Hello\"\n");
            return 1;
        }
        if (!(first.bbox.width > 0 && first.bbox.height > 0)) {
            fprintf(stderr, "assertion failed: first word has no bbox\n");
            return 1;
        }
        printf("WORDS OK\n");
        return 0;
    }
}
