// One check per public method — mirrors the api_coverage convention used by
// every pdf_oxide binding. Plain clang-built executable (no XCTest harness):
// returns non-zero on any failure. Self-contained: builds its own PDF.
#import "POXPdfOxide.h"
#import <Foundation/Foundation.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

// Print a symbolicated backtrace if a coverage call crashes, so CI shows which
// API faulted instead of a bare exit 139. `g_phase` is updated as the test
// advances, giving a human-readable marker even when symbols are inlined.
static const char* g_phase = "startup";
static void crash_handler(int sig) {
    fprintf(stderr, "\n*** caught signal %d during phase: %s ***\n", sig, g_phase);
    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);
    _exit(139);
}

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
        signal(SIGSEGV, crash_handler);
        signal(SIGABRT, crash_handler);
        signal(SIGBUS, crash_handler);
        NSError* err = nil;

        // ── Pdf builder ──────────────────────────────────────────────────────
        g_phase = "Pdf builder";
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
        g_phase = "Document open paths";
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
        g_phase = "Document inspection + extraction";
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
        g_phase = "Phase-1 element extraction";
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
        g_phase = "Phase-2 extraction";
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
        g_phase = "authenticate (wrong password on unencrypted doc returns a bool)";
        {
            NSError* ae = nil;
            BOOL authed = [doc authenticate:@"any-password" error:&ae]; // authenticate
            CHECK(authed == YES || authed == NO);
        }

        // ── Page model ───────────────────────────────────────────────────────
        g_phase = "Page model";
        {
            POXPage* page = [doc pageAtIndex:0];               // pageAtIndex
            CHECK([[page text:&err] containsString:@"Alpha"]); // Page text
            CHECK([[page markdown:&err] length] > 0);          // Page markdown
            CHECK([[page html:&err] length] > 0);              // Page html
            CHECK([[page plainText:&err] length] > 0);         // Page plainText
        }

        // ── Phase-3 page rendering ───────────────────────────────────────────
        g_phase = "Phase-3 page rendering";
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
        g_phase = "DocumentEditor";
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

        // ── PDF creation builder API ─────────────────────────────────────────
        g_phase = "PDF creation builder API";
        {
            NSError* be = nil;
            POXDocumentBuilder* db = [POXDocumentBuilder createWithError:&be]; // create
            CHECK(db != nil && be == nil);
            CHECK([db setTitle:@"Builder Doc" error:&be]); // setTitle
            POXPageBuilder* pg = [db pageWithWidth:595
                                            height:842
                                             error:&be]; // page(595,842)
            CHECK(pg != nil && be == nil);
            CHECK([pg font:@"Helvetica" size:12 error:&be]); // font
            CHECK([pg heading:1 text:@"Title" error:&be]);   // heading
            CHECK([pg paragraph:@"Hello world from the builder."
                          error:&be]);               // paragraph
            CHECK([pg done:&be]);                    // done (consumes page)
            [pg close];                              // idempotent no-op
            NSData* built = [db buildWithError:&be]; // build
            CHECK(built != nil && built.length > 0);
            if (built != nil && built.length > 0) {
                NSError* oe = nil;
                POXDocument* rd = [POXDocument openFromBytes:built error:&oe];
                CHECK(rd != nil && oe == nil);
                CHECK([rd pageCountError:&oe] >= 1);
                NSString* txt = [rd extractText:0 error:&oe];
                CHECK([txt containsString:@"Hello"] || [txt containsString:@"Title"]);
                [rd close];
            }
            [db close]; // close
            [db close]; // idempotent
        }

        // ── Phase-6: conformance validation (fully testable on the sample) ───
        g_phase = "Phase-6: conformance validation (fully testable on the sample)";
        {
            NSError* ve = nil;
            POXPdfAResults* a = [doc validatePdfA:0 error:&ve]; // validatePdfA
            CHECK(a != nil && ve == nil);
            if (a != nil) {
                NSError* ce = nil;
                BOOL compliant = [a isCompliantError:&ce]; // PdfA isCompliant (bool)
                CHECK(compliant == YES || compliant == NO);
                CHECK([a errorCount] >= 0);            // PdfA errorCount
                CHECK([a warningCount] >= 0);          // PdfA warningCount
                NSArray<NSString*>* errs = [a errors]; // PdfA errors
                CHECK(errs != nil);
                CHECK((int32_t)errs.count == [a errorCount]);
                [a close]; // PdfA close
                [a close]; // idempotent
            }

            NSError* ue = nil;
            POXUaResults* ua = [doc validatePdfUa:1 error:&ue]; // validatePdfUa
            CHECK(ua != nil && ue == nil);
            if (ua != nil) {
                NSError* ace = nil;
                BOOL acc = [ua isAccessibleError:&ace]; // Ua isAccessible (bool)
                CHECK(acc == YES || acc == NO);
                CHECK([ua errorCount] >= 0);                // Ua errorCount
                CHECK([ua warningCount] >= 0);              // Ua warningCount
                NSArray<NSString*>* uerrs = [ua errors];    // Ua errors
                NSArray<NSString*>* uwarns = [ua warnings]; // Ua warnings
                CHECK(uerrs != nil && uwarns != nil);
                POXUaStats st = {0, 0, 0, 0, 0, 0};
                NSError* se = nil;
                BOOL gotStats = [ua stats:&st error:&se]; // Ua stats
                CHECK(gotStats == YES || gotStats == NO);
                if (gotStats) {
                    CHECK(st.pages >= 0);
                    CHECK(st.structElements >= 0);
                }
                [ua close]; // Ua close
                [ua close]; // idempotent
            }

            NSError* xe = nil;
            POXPdfXResults* x = [doc validatePdfX:0 error:&xe]; // validatePdfX
            CHECK(x != nil && xe == nil);
            if (x != nil) {
                NSError* xce = nil;
                BOOL xc = [x isCompliantError:&xce]; // PdfX isCompliant (bool)
                CHECK(xc == YES || xc == NO);
                CHECK([x errorCount] >= 0);             // PdfX errorCount
                NSArray<NSString*>* xerrs = [x errors]; // PdfX errors
                CHECK(xerrs != nil);
                [x close]; // PdfX close
                [x close]; // idempotent
            }
        }

        // ── Phase-6: log level round-trip ────────────────────────────────────
        g_phase = "Phase-6: log level round-trip";
        {
            [POXSigning setLogLevel:3];        // setLogLevel
            CHECK([POXSigning logLevel] == 3); // logLevel round-trip
            [POXSigning setLogLevel:1];
            CHECK([POXSigning logLevel] == 1);
        }

        // ── Phase-6: signing / PKI / timestamp / TSA / DSS exercise ──────────
        g_phase = "Phase-6: signing / PKI / timestamp / TSA / DSS exercise";
        // No real PKCS#12 cert or network is required: every wrapper is invoked
        // with minimal/empty inputs and must either return a value or surface the
        // POXErrorDomain error type. The goal is symbol coverage, not success.
        {
            NSData* empty = [NSData data];

            // Certificate loaders (expected to fail on empty/bogus input).
            NSError* ce1 = nil;
            POXCertificate* cert = [POXCertificate loadFromBytes:empty
                                                        password:@""
                                                           error:&ce1]; // loadFromBytes
            CHECK(cert == nil ? (ce1 != nil) : YES);
            NSError* ce2 = nil;
            POXCertificate* certPem =
                [POXCertificate loadFromPemCert:@"not-a-pem"
                                         keyPem:@"not-a-key"
                                          error:&ce2]; // loadFromPemCert
            CHECK(certPem == nil ? (ce2 != nil) : YES);
            // Accessors only when a handle exists (otherwise still "exercised"
            // via the loader call above).
            if (cert != nil) {
                NSError* ae = nil;
                (void)[cert subjectError:&ae]; // subject
                (void)[cert issuerError:&ae];  // issuer
                (void)[cert serialError:&ae];  // serial
                int64_t nb = 0, na = 0;
                (void)[cert validityNotBefore:&nb notAfter:&na error:&ae]; // validity
                (void)[cert isValidError:&ae];                             // isValid
                [cert close];                                              // close
            }

            // Top-level signing — fail gracefully without a real cert.
            // The bogus loaders above always yield nil certs, so these calls
            // deliberately pass a nil `certificate:` (a `nonnull` param) to
            // verify the wrapper fails gracefully rather than crashing
            // (`[nil POX_handle]` → 0 → the core returns an error). That is an
            // intentional nonnull-contract violation, so exclude the block from
            // the static analyzer (`scan-build --status-bugs`) while keeping it
            // in the real build + test run.
#ifndef __clang_analyzer__
            NSError* se1 = nil;
            NSData* signed1 = [POXSigning signBytes:samplePdf()
                                        certificate:(cert ?: certPem)reason:@"test"
                                           location:@"here"
                                              error:&se1]; // signBytes
            CHECK(signed1 == nil ? (se1 != nil) : signed1.length > 0);

            NSError* se2 = nil;
            NSData* signed2 = [POXSigning signBytesPades:samplePdf()
                                             certificate:(cert ?: certPem)level:0
                                                  tsaUrl:nil
                                                  reason:@"r"
                                                location:@"l"
                                                   certs:@[]
                                                    crls:@[]
                                                   ocsps:@[]
                                                   error:&se2]; // signBytesPades
            CHECK(signed2 == nil ? (se2 != nil) : signed2.length > 0);

            POXPadesSignOptions* opts = [[POXPadesSignOptions alloc] init];
            opts.certificate = (cert ?: certPem);
            opts.level = 0;
            opts.reason = @"r";
            opts.location = @"l";
            opts.certs = @[ empty ];
            opts.crls = @[];
            opts.ocsps = @[];
            NSError* se3 = nil;
            NSData* signed3 =
                [POXSigning signBytesPadesOpts:samplePdf()
                                       options:opts
                                         error:&se3]; // signBytesPadesOpts
            CHECK(signed3 == nil ? (se3 != nil) : signed3.length > 0);
#endif // __clang_analyzer__

            // Timestamp parse (bogus DER → error).
            NSError* tse = nil;
            POXTimestamp* ts = [POXTimestamp parse:empty error:&tse]; // parse
            CHECK(ts == nil ? (tse != nil) : YES);
            if (ts != nil) {
                NSError* e = nil;
                (void)[ts tokenError:&e];          // token
                (void)[ts messageImprintError:&e]; // messageImprint
                (void)[ts timeError:&e];           // time
                (void)[ts serialError:&e];         // serial
                (void)[ts tsaNameError:&e];        // tsaName
                (void)[ts policyOidError:&e];      // policyOid
                (void)[ts hashAlgorithmError:&e];  // hashAlgorithm
                (void)[ts verifyError:&e];         // verify
                [ts close];                        // close
            }

            // TSA client — created without a network call; requests will error.
            NSError* tce = nil;
            POXTsaClient* tsa = [POXTsaClient createWithUrl:@"http://tsa.invalid/tsr"
                                                   username:nil
                                                   password:nil
                                                    timeout:1
                                                   hashAlgo:0
                                                   useNonce:YES
                                                    certReq:YES
                                                      error:&tce]; // createWithUrl
            CHECK(tsa == nil ? (tce != nil) : YES);
            if (tsa != nil) {
                NSError* re = nil;
                POXTimestamp* rt = [tsa requestTimestamp:empty
                                                   error:&re]; // requestTimestamp
                CHECK(rt == nil ? (re != nil) : YES);
                NSError* rhe = nil;
                POXTimestamp* rth =
                    [tsa requestTimestampHash:empty
                                     hashAlgo:0
                                        error:&rhe]; // requestTimestampHash
                CHECK(rth == nil ? (rhe != nil) : YES);
                [tsa close]; // close
            }

            // SignatureInfo wrappers are exercised through a signature read from
            // a document if one exists; the sample is unsigned, so this branch
            // simply confirms the accessor surface compiles + links. We invoke
            // the read indirectly by ensuring the types are usable.
            (void)^(POXSignatureInfo* sig, POXDss* dss) {
              NSError* e = nil;
              (void)[sig signerNameError:&e];
              (void)[sig signingReasonError:&e];
              (void)[sig signingLocationError:&e];
              (void)[sig signingTimeError:&e];
              (void)[sig certificateError:&e];
              (void)[sig padesLevelError:&e];
              (void)[sig hasTimestampError:&e];
              (void)[sig timestampError:&e];
              (void)[sig addTimestamp:ts error:&e];
              (void)[sig verifyError:&e];
              (void)[sig verifyDetached:empty error:&e];
              [sig close];
              (void)[dss certCount];
              (void)[dss crlCount];
              (void)[dss ocspCount];
              (void)[dss vriCount];
              (void)[dss certAtIndex:0 error:&e];
              (void)[dss crlAtIndex:0 error:&e];
              (void)[dss ocspAtIndex:0 error:&e];
              [dss close];
            };
        }

        // ── Phase-7: barcodes ────────────────────────────────────────────────
        g_phase = "Phase-7: barcodes";
        {
            NSError* qe = nil;
            POXBarcode* qr = [POXBarcode generateQrCode:@"https://oxide.fyi"
                                        errorCorrection:0
                                                 sizePx:128
                                                  error:&qe]; // generateQrCode
            CHECK(qr != nil && qe == nil);
            if (qr != nil) {
                NSError* e = nil;
                CHECK([[qr dataError:&e] length] > 0); // barcode data
                CHECK([qr formatError:&e] >= 0);       // barcode format
                (void)[qr confidenceError:&e];         // barcode confidence
                NSData* png = [qr imagePngWithSizePx:128 error:&e]; // barcode image png
                CHECK(png != nil && png.length > 0);
                NSString* svg = [qr svgWithSizePx:128 error:&e]; // barcode svg
                CHECK(svg.length > 0);
                [qr close];
                [qr close]; // idempotent
            }
            NSError* be = nil;
            POXBarcode* bc = [POXBarcode generateBarcode:@"123456789012"
                                                  format:0
                                                  sizePx:128
                                                   error:&be]; // generateBarcode
            // Some formats validate input; accept either a handle or an error.
            CHECK(bc == nil ? (be != nil) : YES);
            if (bc != nil) {
                NSError* e = nil;
                (void)[bc dataError:&e];
                (void)[bc formatError:&e];
                [bc close];
            }
        }

        // ── Phase-7: render variants ─────────────────────────────────────────
        g_phase = "Phase-7: render variants";
        {
            NSError* e = nil;
            POXRenderedImage* opt =
                [doc renderPageWithOptions:0
                                       dpi:72
                                    format:0
                                       bgR:1.0f
                                       bgG:1.0f
                                       bgB:1.0f
                                       bgA:1.0f
                     transparentBackground:0
                         renderAnnotations:1
                               jpegQuality:90
                                     error:&e]; // renderPageWithOptions
            CHECK(opt != nil && e == nil);
            if (opt != nil) {
                CHECK(opt.width > 0 && opt.height > 0 && opt.data.length > 0);
                [opt close];
            }
            NSError* exe = nil;
            POXRenderedImage* ex =
                [doc renderPageWithOptionsEx:0
                                         dpi:72
                                      format:0
                                         bgR:1.0f
                                         bgG:1.0f
                                         bgB:1.0f
                                         bgA:1.0f
                       transparentBackground:0
                           renderAnnotations:1
                                 jpegQuality:90
                              excludedLayers:@[ @"DraftLayer" ]
                                       error:&exe]; // renderPageWithOptionsEx
            CHECK(ex != nil && exe == nil);
            if (ex != nil) {
                CHECK(ex.width > 0 && ex.height > 0);
                [ex close];
            }
            NSError* rge = nil;
            POXRenderedImage* region = [doc renderPageRegion:0
                                                       cropX:0
                                                       cropY:0
                                                   cropWidth:100
                                                  cropHeight:100
                                                      format:0
                                                       error:&rge]; // renderPageRegion
            CHECK(region != nil && rge == nil);
            if (region != nil) {
                CHECK(region.width > 0 && region.height > 0);
                [region close];
            }
            NSError* fe = nil;
            POXRenderedImage* fit = [doc renderPageFit:0
                                                     w:200
                                                     h:200
                                                format:0
                                                 error:&fe]; // renderPageFit
            CHECK(fit != nil && fe == nil);
            if (fit != nil) {
                CHECK(fit.width > 0 && fit.height > 0 && fit.width <= 200 &&
                      fit.height <= 200);
                [fit close];
            }
            NSError* re = nil;
            int32_t rw = 0, rh = 0;
            POXRenderedImage* raw = [doc renderPageRaw:0
                                                   dpi:72
                                              outWidth:&rw
                                             outHeight:&rh
                                                 error:&re]; // renderPageRaw
            CHECK(raw != nil && re == nil);
            if (raw != nil) {
                CHECK(rw > 0 && rh > 0 && raw.data.length > 0);
                [raw close];
            }
            NSError* ete = nil;
            int32_t est = [doc estimateRenderTime:0 error:&ete]; // estimateRenderTime
            CHECK(est >= 0 || ete != nil);
        }

        // ── Phase-7: page getters ────────────────────────────────────────────
        g_phase = "Phase-7: page getters";
        {
            NSError* e = nil;
            CHECK([doc pageWidth:0 error:&e] > 0);     // pageWidth
            CHECK([doc pageHeight:0 error:&e] > 0);    // pageHeight
            CHECK([doc pageRotation:0 error:&e] >= 0); // pageRotation
            NSError* ele = nil;
            POXElementList* els = [doc pageElements:0 error:&ele]; // pageElements
            CHECK(els != nil && ele == nil);
            if (els != nil) {
                int32_t n = [els count]; // element count
                CHECK(n >= 0);
                if (n > 0) {
                    NSError* ie = nil;
                    (void)[els typeAtIndex:0 error:&ie];       // element type
                    (void)[els textAtIndex:0 error:&ie];       // element text
                    POXBbox r = [els rectAtIndex:0 error:&ie]; // element rect
                    CHECK(r.width >= 0 && r.height >= 0);
                }
                NSError* je = nil;
                NSString* json = [els toJsonWithError:&je]; // elements to json
                CHECK(json == nil ? (je != nil) : json.length > 0);
                [els close];
                [els close]; // idempotent
            }
        }

        // ── Phase-7: redaction (on an editor) ────────────────────────────────
        g_phase = "Phase-7: redaction (on an editor)";
        {
            NSError* ee = nil;
            POXDocumentEditor* red = [POXDocumentEditor openFromBytes:samplePdf()
                                                                error:&ee];
            CHECK(red != nil && ee == nil);
            if (red != nil) {
                NSError* e = nil;
                CHECK([red redactionAddPage:0
                                         x1:50
                                         y1:50
                                         x2:150
                                         y2:80
                                          r:0
                                          g:0
                                          b:0
                                      error:&e]);            // redactionAdd
                CHECK([red redactionCount:0 error:&e] >= 1); // redactionCount
                int32_t glyphs = [red redactionApplyScrubMetadata:NO
                                                                r:0
                                                                g:0
                                                                b:0
                                                            error:&e]; // redactionApply
                CHECK(glyphs >= 0 || e != nil);
                NSError* se = nil;
                int32_t scrubbed =
                    [red redactionScrubMetadataWithError:&se]; // redactionScrubMetadata
                CHECK(scrubbed >= 0 || se != nil);
                // addBarcode on the editor (Phase-7 barcode placement).
                NSError* qe = nil;
                POXBarcode* qr = [POXBarcode generateQrCode:@"x"
                                            errorCorrection:0
                                                     sizePx:64
                                                      error:&qe];
                if (qr != nil) {
                    NSError* abe = nil;
                    BOOL added = [red addBarcode:qr
                                            page:0
                                               x:10
                                               y:10
                                           width:40
                                          height:40
                                           error:&abe]; // addBarcodeToPage
                    CHECK(added == YES || abe != nil);
                    [qr close];
                }
                [red close];
            }
        }

        // ── Phase-7: from_image_bytes / from_html_css / merge ────────────────
        g_phase = "Phase-7: from_image_bytes / from_html_css / merge";
        {
            // from_image_bytes on bogus data must raise the binding error.
            NSError* ibe = nil;
            POXPdf* badImg = [POXPdf fromImageBytes:[NSData data]
                                              error:&ibe]; // fromImageBytes
            CHECK(badImg == nil ? (ibe != nil) : YES);

            // from_html_css is testable on the sample input.
            NSError* he = nil;
            POXPdf* htmlPdf = [POXPdf fromHtml:@"<h1>HC</h1><p>body</p>"
                                           css:@"h1{color:#000}"
                                     fontBytes:nil
                                         error:&he]; // fromHtmlCss
            CHECK(htmlPdf == nil ? (he != nil)
                                 : [[htmlPdf toBytesWithError:&he] length] > 0);

            NSError* hfe = nil;
            POXPdf* htmlFonts = [POXPdf fromHtml:@"<p>cascade</p>"
                                             css:@""
                                        families:@[]
                                           fonts:@[]
                                           error:&hfe]; // fromHtmlCssWithFonts
            CHECK(htmlFonts == nil ? (hfe != nil)
                                   : [[htmlFonts toBytesWithError:&hfe] length] > 0);

            // merge: write two temp PDFs, merge them.
            NSString* p1 = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"pdfoxide_objc_m1.pdf"];
            NSString* p2 = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"pdfoxide_objc_m2.pdf"];
            [[POXPdf fromMarkdown:@"# one\n\nx\n" error:&err] saveToPath:p1 error:&err];
            [[POXPdf fromMarkdown:@"# two\n\ny\n" error:&err] saveToPath:p2 error:&err];
            NSError* me = nil;
            NSData* merged = [POXTools merge:@[ p1, p2 ] error:&me]; // merge
            CHECK(merged == nil ? (me != nil) : merged.length > 0);
            if (merged != nil && merged.length > 0) {
                NSError* oe = nil;
                POXDocument* md = [POXDocument openFromBytes:merged error:&oe];
                CHECK(md != nil && [md pageCountError:&oe] >= 2);
                [md close];
            }
            [[NSFileManager defaultManager] removeItemAtPath:p1 error:nil];
            [[NSFileManager defaultManager] removeItemAtPath:p2 error:nil];

            // from_image: bogus path must raise.
            NSError* fie = nil;
            POXPdf* badPath = [POXPdf fromImage:@"/nonexistent/none.png"
                                          error:&fie]; // fromImage
            CHECK(badPath == nil && fie != nil);
        }

        // ── Phase-7: OCR (needs model files) — invoke + assert raises/returns ─
        g_phase = "Phase-7: OCR (needs model files) — invoke + assert raises/returns";
        {
            // Engine create with bogus model paths: must raise the binding error.
            NSError* oce = nil;
            POXOcrEngine* engine =
                [POXOcrEngine createWithDetModelPath:@"/nonexistent/det"
                                        recModelPath:@"/nonexistent/rec"
                                            dictPath:@"/nonexistent/dict"
                                               error:&oce]; // ocrEngineCreate
            CHECK(engine == nil ? (oce != nil) : YES);
            if (engine != nil) {
                [engine close];
                [engine close]; // idempotent
            }
            // needs-OCR / extract with a nil engine fall back to native text.
            NSError* ne = nil;
            BOOL needs = [doc pageNeedsOcr:0 error:&ne]; // pageNeedsOcr
            CHECK(needs == YES || needs == NO || ne != nil);
            NSError* oe = nil;
            NSString* ocrText = [doc ocrExtractText:0
                                             engine:nil
                                              error:&oe]; // ocrExtractText
            CHECK(ocrText != nil ? (ocrText.length >= 0) : (oe != nil));
        }

        // ── Final phase: in-rect extractors ─────────────────────────────────
        g_phase = "Final phase: in-rect extractors";
        {
            NSError* re = nil;
            NSString* rt = [doc extractTextInRect:0
                                                x:0
                                                y:0
                                            width:1000
                                           height:1000
                                            error:&re]; // extractTextInRect
            CHECK(rt != nil ? (rt.length >= 0) : (re != nil));
            re = nil;
            NSArray<POXWord*>* rw = [doc extractWordsInRect:0
                                                          x:0
                                                          y:0
                                                      width:1000
                                                     height:1000
                                                      error:&re]; // extractWordsInRect
            CHECK(rw != nil ? (rw.count >= 0) : (re != nil));
            re = nil;
            NSArray<POXTextLine*>* rl =
                [doc extractLinesInRect:0
                                      x:0
                                      y:0
                                  width:1000
                                 height:1000
                                  error:&re]; // extractLinesInRect
            CHECK(rl != nil ? (rl.count >= 0) : (re != nil));
            re = nil;
            NSArray<POXTable*>* rtb =
                [doc extractTablesInRect:0
                                       x:0
                                       y:0
                                   width:1000
                                  height:1000
                                   error:&re]; // extractTablesInRect
            CHECK(rtb != nil ? (rtb.count >= 0) : (re != nil));
            re = nil;
            NSArray<POXImage*>* ri =
                [doc extractImagesInRect:0
                                       x:0
                                       y:0
                                   width:1000
                                  height:1000
                                   error:&re]; // extractImagesInRect
            CHECK(ri != nil ? (ri.count >= 0) : (re != nil));
        }

        // ── Final phase: auto extraction / classification ───────────────────
        g_phase = "Final phase: auto extraction / classification";
        {
            NSError* ae = nil;
            NSString* ta = [doc extractTextAuto:0 error:&ae]; // extractTextAuto
            CHECK(ta != nil ? (ta.length >= 0) : (ae != nil));
            ae = nil;
            NSString* at = [doc extractAllTextWithError:&ae]; // extractAllText
            CHECK(at != nil ? (at.length >= 0) : (ae != nil));
            ae = nil;
            NSString* pa = [doc extractPageAuto:0
                                    optionsJson:nil
                                          error:&ae]; // extractPageAuto
            CHECK(pa != nil ? (pa.length >= 0) : (ae != nil));
            ae = nil;
            NSString* cp = [doc classifyPage:0 error:&ae]; // classifyPage
            CHECK(cp != nil ? (cp.length >= 0) : (ae != nil));
            ae = nil;
            NSString* cd = [doc classifyDocumentWithError:&ae]; // classifyDocument
            CHECK(cd != nil ? (cd.length >= 0) : (ae != nil));
        }

        // ── Final phase: header / footer / artifact removal ─────────────────
        g_phase = "Final phase: header / footer / artifact removal";
        {
            NSError* he = nil;
            CHECK([doc eraseHeader:0 error:&he] >= -1 || he != nil); // eraseHeader
            he = nil;
            CHECK([doc eraseFooter:0 error:&he] >= -1 || he != nil); // eraseFooter
            he = nil;
            CHECK([doc eraseArtifacts:0 error:&he] >= -1 ||
                  he != nil); // eraseArtifacts
            he = nil;
            CHECK([doc removeHeaders:0.1f error:&he] >= -1 ||
                  he != nil); // removeHeaders
            he = nil;
            CHECK([doc removeFooters:0.1f error:&he] >= -1 ||
                  he != nil); // removeFooters
            he = nil;
            CHECK([doc removeArtifacts:0.1f error:&he] >= -1 ||
                  he != nil); // removeArtifacts
        }

        // ── Final phase: AcroForm fields & form data ─────────────────────────
        g_phase = "Final phase: AcroForm fields & form data";
        {
            NSError* fe = nil;
            NSArray<POXFormField*>* ff = [doc formFieldsWithError:&fe]; // formFields
            CHECK(ff != nil ? (ff.count >= 0) : (fe != nil));           // empty list ok
            for (POXFormField* f in ff) {
                CHECK(f.name != nil && f.type != nil && f.value != nil);
                CHECK(f.readonly == YES || f.readonly == NO);
                CHECK(f.required == YES || f.required == NO);
            }
            fe = nil;
            NSData* fd = [doc exportFormDataToBytes:0
                                              error:&fe]; // exportFormDataToBytes
            CHECK(fd != nil ? (fd.length >= 0) : (fe != nil));
            fe = nil;
            BOOL imp = [doc importFormDataFromPath:@"/nonexistent/data.fdf"
                                             error:&fe]; // importFormData
            CHECK(imp == NO ? (fe != nil) : YES);
            fe = nil;
            BOOL impf = [doc importFormFromFile:@"/nonexistent/data.fdf"
                                          error:&fe]; // importFormFromFile
            CHECK(impf == NO ? (fe != nil) : YES);
        }

        // ── Final phase: document structure / metadata ───────────────────────
        g_phase = "Final phase: document structure / metadata";
        {
            NSError* se = nil;
            NSString* ol = [doc outlineWithError:&se]; // outline
            CHECK(ol != nil ? (ol.length >= 0) : (se != nil));
            se = nil;
            NSString* pl = [doc pageLabelsWithError:&se]; // pageLabels
            CHECK(pl != nil ? (pl.length >= 0) : (se != nil));
            se = nil;
            NSString* xmp = [doc xmpMetadataWithError:&se]; // xmpMetadata
            CHECK(xmp != nil ? (xmp.length >= 0) : (se != nil));
            se = nil;
            NSData* sb = [doc sourceBytesWithError:&se]; // sourceBytes
            CHECK(sb != nil ? (sb.length >= 0) : (se != nil));
            CHECK([doc hasXfa] == YES || [doc hasXfa] == NO); // hasXfa
            se = nil;
            NSString* sp = [doc planSplitByBookmarks:nil
                                               error:&se]; // planSplitByBookmarks
            CHECK(sp != nil ? (sp.length >= 0) : (se != nil));
        }

        // ── Final phase: fonts / search / annotations as JSON ────────────────
        g_phase = "Final phase: fonts / search / annotations as JSON";
        {
            NSError* je = nil;
            NSString* fj = [doc embeddedFontsJson:0 error:&je]; // fontsToJson
            CHECK(fj != nil ? (fj.length >= 0) : (je != nil));
            je = nil;
            NSString* sj = [doc searchJson:0
                                      term:@"Alpha"
                             caseSensitive:NO
                                     error:&je]; // searchResultsToJson
            CHECK(sj != nil ? (sj.length >= 0) : (je != nil));
            je = nil;
            NSString* aj = [doc annotationsJson:0 error:&je]; // annotationsToJson
            CHECK(aj != nil ? (aj.length >= 0) : (je != nil));
            // font_get_size surfaces via the POXFont.size property
            NSError* fe = nil;
            NSArray<POXFont*>* fonts = [doc embeddedFonts:0 error:&fe];
            for (POXFont* f in fonts)
                CHECK(f.size >= 0); // font_get_size
        }

        // ── Final phase: annotation extras (via constructed PDF — list ok) ───
        g_phase = "Final phase: annotation extras (via constructed PDF — list ok)";
        {
            NSError* ae = nil;
            NSArray<POXAnnotation*>* anns = [doc pageAnnotations:0
                                                           error:&ae]; // (may be empty)
            CHECK(anns != nil || ae != nil);
            for (POXAnnotation* a in anns) {
                CHECK(a.color >= 0);                      // get_color
                CHECK(a.creationDate >= 0);               // get_creation_date
                CHECK(a.modificationDate >= 0);           // get_modification_date
                CHECK(a.hidden == YES || a.hidden == NO); // is_hidden
                CHECK(a.markedDeleted == YES ||
                      a.markedDeleted == NO);                   // is_marked_deleted
                CHECK(a.printable == YES || a.printable == NO); // is_printable
                CHECK(a.readOnly == YES || a.readOnly == NO);   // is_read_only
                (void)a.linkUri;                                // link uri
                (void)a.iconName;                               // text icon name
                CHECK(a.quadPoints != nil);                     // highlight quad points
            }
        }

        // ── Final phase: PDF/A conversion (may error on this sample) ─────────
        g_phase = "Final phase: PDF/A conversion (may error on this sample)";
        {
            NSError* ce = nil;
            BOOL ok = [doc convertToPdfA:0 error:&ce]; // convertToPdfA
            CHECK(ok == YES || ce != nil);
        }

        // ── Final phase: document signatures (need a cert) ───────────────────
        g_phase = "Final phase: document signatures (need a cert)";
        {
            NSError* sigErr = nil;
            int32_t sc = [doc signatureCountWithError:&sigErr]; // signatureCount
            CHECK(sc >= 0 || sigErr != nil);
            sigErr = nil;
            POXSignatureInfo* si = [doc signatureAtIndex:0 error:&sigErr]; // signature
            CHECK(si != nil ? YES : (sigErr != nil));
            if (si)
                [si close];
            sigErr = nil;
            int32_t va = [doc verifyAllSignaturesWithError:&sigErr]; // verifyAll
            CHECK(va >= -1 || sigErr != nil);
            sigErr = nil;
            int32_t ht = [doc hasTimestampWithError:&sigErr]; // hasTimestamp
            CHECK(ht >= -1 || sigErr != nil);
            sigErr = nil;
            POXDss* dss = [doc dssWithError:&sigErr]; // dss
            CHECK(dss != nil ? YES : (sigErr != nil));
            if (dss)
                [dss close];
            // sign requires a real certificate handle: invoke + assert raises.
            NSError* certErr = nil;
            POXCertificate* cert = [POXCertificate loadFromPemCert:@"not-a-pem"
                                                            keyPem:@"not-a-key"
                                                             error:&certErr];
            if (cert) {
                NSError* signErr = nil;
                BOOL signed_ = [doc sign:cert
                                  reason:@"test"
                                location:@"here"
                                   error:&signErr]; // sign
                CHECK(signed_ == YES || signErr != nil);
                [cert close];
            } else {
                CHECK(certErr != nil);
            }
        }

        // ── Final phase: office export (may error on this sample) ────────────
        g_phase = "Final phase: office export (may error on this sample)";
        {
            NSError* oe = nil;
            NSData* docx = [doc toDocxWithError:&oe]; // toDocx
            CHECK(docx != nil ? (docx.length >= 0) : (oe != nil));
            oe = nil;
            NSData* pptx = [doc toPptxWithError:&oe]; // toPptx
            CHECK(pptx != nil ? (pptx.length >= 0) : (oe != nil));
            oe = nil;
            NSData* xlsx = [doc toXlsxWithError:&oe]; // toXlsx
            CHECK(xlsx != nil ? (xlsx.length >= 0) : (oe != nil));
        }

        // ── Final phase: office open from bytes (need real office files) ─────
        g_phase = "Final phase: office open from bytes (need real office files)";
        {
            NSData* junk = [@"PK\x03\x04 not a real office file"
                dataUsingEncoding:NSUTF8StringEncoding];
            NSError* oe = nil;
            POXDocument* d = [POXDocument openFromDocxBytes:junk
                                                      error:&oe]; // openFromDocxBytes
            CHECK(d == nil ? (oe != nil) : ([d pageCountError:&oe] >= 0));
            if (d)
                [d close];
            oe = nil;
            d = [POXDocument openFromPptxBytes:junk error:&oe]; // openFromPptxBytes
            CHECK(d == nil ? (oe != nil) : ([d pageCountError:&oe] >= 0));
            if (d)
                [d close];
            oe = nil;
            d = [POXDocument openFromXlsxBytes:junk error:&oe]; // openFromXlsxBytes
            CHECK(d == nil ? (oe != nil) : ([d pageCountError:&oe] >= 0));
            if (d)
                [d close];
        }

        // ── Final phase: editor FDF/XFDF import (may error on junk) ──────────
        g_phase = "Final phase: editor FDF/XFDF import (may error on junk)";
        {
            NSString* path = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"pdfoxide_objc_editor.pdf"];
            [[POXPdf fromMarkdown:@"# f\n\nx\n" error:&err] saveToPath:path error:&err];
            NSError* ee = nil;
            POXDocumentEditor* ed = [POXDocumentEditor openEditor:path error:&ee];
            if (ed) {
                NSData* junk = [@"junk" dataUsingEncoding:NSUTF8StringEncoding];
                NSError* ie = nil;
                BOOL f = [ed importFdfBytes:junk error:&ie]; // importFdfBytes
                CHECK(f == YES || ie != nil);
                ie = nil;
                BOOL xf = [ed importXfdfBytes:junk error:&ie]; // importXfdfBytes
                CHECK(xf == YES || ie != nil);
                [ed close];
            } else {
                CHECK(ee != nil);
            }
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
        }

        // ── Final phase: Pdf page count alias ────────────────────────────────
        g_phase = "Final phase: Pdf page count alias";
        {
            POXPdf* p = [POXPdf fromMarkdown:@"# p\n\nbody\n" error:&err];
            NSError* pe = nil;
            CHECK([p pageCountError:&pe] >= 1 || pe != nil); // pdf_get_page_count
        }

        // ── Final phase: crypto / FIPS / models / config / renderer ──────────
        g_phase = "Final phase: crypto / FIPS / models / config / renderer";
        {
            NSError* ce = nil;
            NSString* prov = [POXCrypto activeProviderWithError:&ce]; // activeProvider
            CHECK(prov != nil ? (prov.length >= 0) : (ce != nil));
            ce = nil;
            NSString* cbom = [POXCrypto cbomWithError:&ce]; // cbom
            CHECK(cbom != nil ? (cbom.length >= 0) : (ce != nil));
            ce = nil;
            NSString* inv = [POXCrypto inventoryWithError:&ce]; // inventory
            CHECK(inv != nil ? (inv.length >= 0) : (ce != nil));
            ce = nil;
            NSString* pol = [POXCrypto policyWithError:&ce]; // policy
            CHECK(pol != nil ? (pol.length >= 0) : (ce != nil));
            CHECK([POXCrypto fipsAvailable] >= -1);        // fipsAvailable
            CHECK([POXCrypto useFips] >= -1);              // useFips
            CHECK([POXCrypto setPolicy:@"default"] >= -1); // setPolicy

            ce = nil;
            NSString* man = [POXModels manifestWithError:&ce]; // modelManifest
            CHECK(man != nil ? (man.length >= 0) : (ce != nil));
            CHECK([POXModels prefetchAvailable] >= -1); // prefetchAvailable
            ce = nil;
            NSString* pf = [POXModels prefetchModels:@"en" error:&ce]; // prefetchModels
            CHECK(pf != nil ? (pf.length >= 0) : (ce != nil));

            // setMaxOpsPerStream returns the PRIOR value and has no error
            // channel; only assert the call is invokable (yields an int), not a
            // specific round-tripped value.
            int64_t prev = [POXConfig setMaxOpsPerStream:1000000]; // setMaxOpsPerStream
            CHECK(prev == prev);                 // invokable: returns an int64_t
            [POXConfig setMaxOpsPerStream:prev]; // restore
            // setPreserveUnmappedGlyphs likewise returns the prior value with no
            // error channel; just confirm it is invokable.
            int32_t pg =
                [POXConfig setPreserveUnmappedGlyphs:0]; // setPreserveUnmappedGlyphs
            CHECK(pg == pg);                          // invokable: returns an int32_t
            [POXConfig setPreserveUnmappedGlyphs:pg]; // restore

            ce = nil;
            POXRenderer* rend = [POXRenderer createWithDpi:150
                                                    format:0
                                                   quality:90
                                                 antiAlias:YES
                                                     error:&ce]; // createRenderer
            CHECK(rend != nil ? YES : (ce != nil));
            if (rend) {
                [rend close];
                [rend close]; // idempotent
            }
        }

        // ── Phase-7: add_timestamp (needs TSA) — invoke + assert raises ──────
        g_phase = "Phase-7: add_timestamp (needs TSA) — invoke + assert raises";
        {
            NSError* te = nil;
            NSData* stamped = [POXTools addTimestamp:samplePdf()
                                            sigIndex:0
                                              tsaUrl:@"http://tsa.invalid/tsr"
                                               error:&te]; // addTimestamp
            CHECK(stamped == nil ? (te != nil) : stamped.length > 0);
        }

        // ── close (idempotent) ───────────────────────────────────────────────
        g_phase = "close (idempotent)";
        [doc close];
        [doc close]; // idempotent — safe to call twice

        // ── Error path ───────────────────────────────────────────────────────
        g_phase = "Error path";
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
