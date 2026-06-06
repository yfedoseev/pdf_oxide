// 01 — Extract text (Java)
//
// Opens a PDF, prints the page count, then the text of each page.
//
// Compile + run against the built jar (which embeds the JNI native lib):
//   javac -cp pdf_oxide.jar ExtractText.java
//   java  -cp pdf_oxide.jar:. ExtractText ../../../tests/fixtures/simple.pdf

import fyi.oxide.pdf.PdfDocument;

public final class ExtractText {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("usage: java ExtractText <pdf>");
            System.exit(1);
        }
        try (PdfDocument doc = PdfDocument.open(args[0])) {
            int pages = doc.pageCount();
            System.out.println("Pages: " + pages);
            for (int i = 0; i < pages; i++) {
                System.out.println("--- Page " + (i + 1) + " ---");
                System.out.println(doc.extractText(i));
            }
        }
    }
}
