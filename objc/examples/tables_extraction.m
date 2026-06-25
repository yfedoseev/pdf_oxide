// tables_extraction — build a PDF with a Markdown table, then extract tables.
// Shared-scenario regression example. Synthetic docs may yield 0 tables; the
// call must merely succeed (no error). Returns non-zero on assertion failure.
#import "POXPdfOxide.h"
#import <Foundation/Foundation.h>

int main(void) {
    @autoreleasepool {
        NSError* err = nil;
        POXPdf* pdf =
            [POXPdf fromMarkdown:@"# Report\n\n| Name | Value |\n|------|-------|\n| "
                                 @"alpha | 1 |\n| beta | 2 |\n"
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

        NSError* te = nil;
        NSArray<POXTable*>* tables = [doc extractTables:0 error:&te];
        if (tables == nil || te != nil) {
            fprintf(stderr, "assertion failed: extractTables returned an error\n");
            return 1;
        }

        printf("tables: %lu\n", (unsigned long)tables.count);
        for (NSUInteger ti = 0; ti < tables.count; ++ti) {
            POXTable* table = tables[ti];
            printf("table %lu: rows=%ld cols=%ld hasHeader=%d\n", (unsigned long)ti,
                   (long)table.rowCount, (long)table.colCount, table.hasHeader);
            if (table.rowCount > 0 && table.colCount > 0) {
                NSString* cell = [table cellTextAtRow:0 col:0];
                printf("  cell(0,0): \"%s\"\n", cell.UTF8String);
            }
        }

        printf("TABLES OK\n");
        return 0;
    }
}
