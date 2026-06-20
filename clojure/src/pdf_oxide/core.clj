;; pdf_oxide — idiomatic Clojure wrapper over the `fyi.oxide:pdf-oxide` Java
;; binding (which owns the single JNI native bridge). No JNA / native code: we
;; call the Java classes directly via interop, returning Clojure-friendly values
;; (java.util.List -> vector, java.util.Optional -> value-or-nil). The handle
;; types (Pdf, PdfDocument, DocumentEditor) implement AutoCloseable, so use
;; `with-open` for deterministic cleanup.
(ns pdf-oxide.core
  (:refer-clojure :exclude [chars]) ; pdf/chars (page glyphs) intentionally shadows clojure.core/chars
  (:import [fyi.oxide.pdf Pdf PdfDocument PdfPage DocumentEditor AutoExtractor]
           [java.util Optional]))

(defn- opt->nil
  "java.util.Optional -> value, or nil when empty."
  [^Optional o]
  (when (and o (.isPresent o)) (.get o)))

;; ── Creation ────────────────────────────────────────────────────────────────
(defn from-markdown ^Pdf [^String markdown] (Pdf/fromMarkdown markdown))
(defn from-html ^Pdf [^String html] (Pdf/fromHtml html))
(defn save
  "Serialize a built Pdf to a byte array."
  ^bytes [^Pdf pdf] (.save pdf))

;; ── Opening ───────────────────────────────────────────────────────────────--
(defn open
  "Open a document from a byte array or a filesystem path string (optional password)."
  (^PdfDocument [source]
   (if (bytes? source) (PdfDocument/open ^bytes source) (PdfDocument/open ^String source)))
  (^PdfDocument [source ^String password]
   (if (bytes? source)
     (PdfDocument/open ^bytes source password)
     (PdfDocument/open ^String source password))))

;; ── Document queries / extraction ─────────────────────────────────────────--
(defn page-count [^PdfDocument doc] (.pageCount doc))
(defn authenticate [^PdfDocument doc ^String password] (.authenticate doc password))
(defn extract-text [^PdfDocument doc page] (.extractText doc (int page)))
(defn to-markdown
  ([^PdfDocument doc] (.toMarkdown doc))
  ([^PdfDocument doc page] (.toMarkdown doc (int page))))
(defn to-html
  ([^PdfDocument doc] (.toHtml doc))
  ([^PdfDocument doc page] (.toHtml doc (int page))))
(defn extract-structured [^PdfDocument doc page] (.extractStructured doc (int page)))
(defn render
  (^bytes [^PdfDocument doc page] (.render doc (int page)))
  (^bytes [^PdfDocument doc page dpi] (.render doc (int page) (int dpi))))
(defn producer [^PdfDocument doc] (opt->nil (.producer doc)))
(defn creator [^PdfDocument doc] (opt->nil (.creator doc)))
(defn form-fields [^PdfDocument doc] (vec (.formFields doc)))
(defn search
  "Search the document; returns a vector of SearchMatch."
  [^PdfDocument doc ^String query] (vec (.search doc query)))
(defn page ^PdfPage [^PdfDocument doc idx] (.page doc (int idx)))
(defn pages [^PdfDocument doc] (vec (.pages doc)))

;; ── Page element extraction (List -> vector) ─────────────────────────────────
(defn words [^PdfPage page] (vec (.words page)))
(defn lines [^PdfPage page] (vec (.lines page)))
(defn chars [^PdfPage page] (vec (.chars page)))
(defn tables [^PdfPage page] (vec (.tables page)))
(defn images [^PdfPage page] (vec (.images page)))
(defn annotations [^PdfPage page] (vec (.annotations page)))
(defn page-text
  ([^PdfPage page] (.text page))
  ([^PdfPage page region] (.text page region)))

;; ── Editing ──────────────────────────────────────────────────────────────--
(defn editor
  "Open a DocumentEditor from a byte array or path string."
  ^DocumentEditor [source]
  (if (bytes? source) (DocumentEditor/open ^bytes source) (DocumentEditor/open ^String source)))
(defn scrub-metadata [^DocumentEditor ed] (.scrubMetadata ed))
(defn add-redaction [^DocumentEditor ed page region] (.addRedaction ed (int page) region))
(defn apply-redactions [^DocumentEditor ed] (.applyRedactionsDestructive ed))
(defn editor-save ^bytes [^DocumentEditor ed] (.save ed))

;; ── Auto extraction ─────────────────────────────────────────────────────────
(defn auto-extractor ^AutoExtractor [^PdfDocument doc] (AutoExtractor/of doc))
(defn auto-text [^AutoExtractor ax] (.extractText ax))

;; ── Lifecycle (prefer `with-open`; these are escape hatches) ─────────────────
(defn close [resource] (.close ^java.lang.AutoCloseable resource))
(defn open? [resource] (.isOpen resource))
