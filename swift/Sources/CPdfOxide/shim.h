/* Re-export the pdf_oxide C ABI header for the Swift system-library module.
 * The actual header lives in the repo's include/ dir, added to the search path
 * via -I in Package.swift (PDF_OXIDE_INCLUDE_DIR). */
#include <pdf_oxide_c/pdf_oxide.h>
