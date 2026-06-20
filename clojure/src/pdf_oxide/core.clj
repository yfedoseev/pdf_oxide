;; pdf_oxide — idiomatic Clojure bindings over the C ABI via JNA.
;;
;; JNA loads the cdylib (libpdf_oxide) by name. Document/pdf handles are opaque
;; records wrapping a JNA Pointer; returned C strings/buffers are copied into
;; Clojure and freed via free_string; non-success C-ABI error codes throw
;; ex-info with {:code …}. Handles implement java.io.Closeable for `with-open`.
;;
;; API surface mirrors the other language bindings; coverage is asserted by
;; pdf-oxide.core-test (one test per public fn).
(ns pdf-oxide.core
  (:import [com.sun.jna Native Library Pointer]
           [com.sun.jna.ptr IntByReference ByteByReference]
           [java.io Closeable]))

;; ── raw JNA binding ───────────────────────────────────────────────────────────
(definterface CLib
  (^com.sun.jna.Pointer pdf_document_open [^String path ^com.sun.jna.ptr.IntByReference code])
  (^com.sun.jna.Pointer pdf_document_open_from_bytes [^bytes data ^long len ^com.sun.jna.ptr.IntByReference code])
  (^com.sun.jna.Pointer pdf_document_open_with_password [^String path ^String pw ^com.sun.jna.ptr.IntByReference code])
  (^void pdf_document_free [^com.sun.jna.Pointer h])
  (^int pdf_document_get_page_count [^com.sun.jna.Pointer h ^com.sun.jna.ptr.IntByReference code])
  (^void pdf_document_get_version [^com.sun.jna.Pointer h ^com.sun.jna.ptr.ByteByReference major ^com.sun.jna.ptr.ByteByReference minor])
  (^boolean pdf_document_is_encrypted [^com.sun.jna.Pointer h])
  (^boolean pdf_document_has_structure_tree [^com.sun.jna.Pointer h])
  (^com.sun.jna.Pointer pdf_document_extract_text [^com.sun.jna.Pointer h ^int page ^com.sun.jna.ptr.IntByReference code])
  (^com.sun.jna.Pointer pdf_document_to_plain_text [^com.sun.jna.Pointer h ^int page ^com.sun.jna.ptr.IntByReference code])
  (^com.sun.jna.Pointer pdf_document_to_markdown [^com.sun.jna.Pointer h ^int page ^com.sun.jna.ptr.IntByReference code])
  (^com.sun.jna.Pointer pdf_document_to_html [^com.sun.jna.Pointer h ^int page ^com.sun.jna.ptr.IntByReference code])
  (^com.sun.jna.Pointer pdf_document_to_markdown_all [^com.sun.jna.Pointer h ^com.sun.jna.ptr.IntByReference code])
  (^com.sun.jna.Pointer pdf_document_extract_structured_to_json [^com.sun.jna.Pointer h ^int page ^com.sun.jna.ptr.IntByReference code])
  (^com.sun.jna.Pointer pdf_from_markdown [^String md ^com.sun.jna.ptr.IntByReference code])
  (^com.sun.jna.Pointer pdf_from_html [^String html ^com.sun.jna.ptr.IntByReference code])
  (^com.sun.jna.Pointer pdf_from_text [^String text ^com.sun.jna.ptr.IntByReference code])
  (^void pdf_free [^com.sun.jna.Pointer h])
  (^int pdf_save [^com.sun.jna.Pointer h ^String path ^com.sun.jna.ptr.IntByReference code])
  (^com.sun.jna.Pointer pdf_save_to_bytes [^com.sun.jna.Pointer h ^com.sun.jna.ptr.IntByReference len ^com.sun.jna.ptr.IntByReference code])
  (^void free_string [^com.sun.jna.Pointer p]))

(def ^CLib lib (Native/load "pdf_oxide" CLib))

(defn- take-string [^Pointer p ^long code op]
  (when (nil? p) (throw (ex-info (str "pdf_oxide: " op " failed") {:code code :op op})))
  (let [s (.getString p 0)] (.free_string lib p) s))

;; ── Document ──────────────────────────────────────────────────────────────────
(deftype Document [^:volatile-mutable ^Pointer handle]
  Closeable
  (close [_] (when handle (.pdf_document_free lib handle))))

(defn- doc-ptr ^Pointer [^Document d]
  (or (.handle d) (throw (ex-info "PdfDocument is closed" {}))))

(defn open
  "Open a PDF from a path. Optional :password."
  [path & {:keys [password]}]
  (let [code (IntByReference.)
        h (if password
            (.pdf_document_open_with_password lib path password code)
            (.pdf_document_open lib path code))]
    (when (nil? h) (throw (ex-info "pdf_oxide: open failed" {:code (.getValue code) :op "open"})))
    (Document. h)))

(defn open-bytes
  "Open a PDF from a byte array."
  [^bytes data]
  (let [code (IntByReference.)
        h (.pdf_document_open_from_bytes lib data (long (alength data)) code)]
    (when (nil? h) (throw (ex-info "pdf_oxide: open-bytes failed" {:code (.getValue code) :op "open-bytes"})))
    (Document. h)))

(defn page-count [^Document d]
  (let [code (IntByReference.) n (.pdf_document_get_page_count lib (doc-ptr d) code)]
    (when (neg? n) (throw (ex-info "pdf_oxide: page-count failed" {:code (.getValue code)})))
    n))

(defn version
  "PDF version as [major minor]."
  [^Document d]
  (let [maj (ByteByReference.) min (ByteByReference.)]
    (.pdf_document_get_version lib (doc-ptr d) maj min)
    [(bit-and (.getValue maj) 0xff) (bit-and (.getValue min) 0xff)]))

(defn encrypted? [^Document d] (.pdf_document_is_encrypted lib (doc-ptr d)))
(defn structure-tree? [^Document d] (.pdf_document_has_structure_tree lib (doc-ptr d)))

(defn- str-page [f ^Document d ^long page op]
  (let [code (IntByReference.)] (take-string (f lib (doc-ptr d) (int page) code) (.getValue code) op)))

(defn extract-text [^Document d page] (str-page #(.pdf_document_extract_text %1 %2 %3 %4) d page "extract-text"))
(defn to-plain-text [^Document d page] (str-page #(.pdf_document_to_plain_text %1 %2 %3 %4) d page "to-plain-text"))
(defn to-markdown [^Document d page] (str-page #(.pdf_document_to_markdown %1 %2 %3 %4) d page "to-markdown"))
(defn to-html [^Document d page] (str-page #(.pdf_document_to_html %1 %2 %3 %4) d page "to-html"))
(defn extract-structured-json [^Document d page]
  (str-page #(.pdf_document_extract_structured_to_json %1 %2 %3 %4) d page "extract-structured-json"))

(defn to-markdown-all [^Document d]
  (let [code (IntByReference.)]
    (take-string (.pdf_document_to_markdown_all lib (doc-ptr d) code) (.getValue code) "to-markdown-all")))

;; ── Pdf builder ───────────────────────────────────────────────────────────────
(deftype Pdf [^:volatile-mutable ^Pointer handle]
  Closeable
  (close [_] (when handle (.pdf_free lib handle))))

(defn- pdf-ptr ^Pointer [^Pdf p]
  (or (.handle p) (throw (ex-info "Pdf is closed" {}))))

(defn- build [f input op]
  (let [code (IntByReference.) h (f lib input code)]
    (when (nil? h) (throw (ex-info (str "pdf_oxide: " op " failed") {:code (.getValue code) :op op})))
    (Pdf. h)))

(defn from-markdown [md] (build #(.pdf_from_markdown %1 %2 %3) md "from-markdown"))
(defn from-html [html] (build #(.pdf_from_html %1 %2 %3) html "from-html"))
(defn from-text [text] (build #(.pdf_from_text %1 %2 %3) text "from-text"))

(defn save [^Pdf p path]
  (let [code (IntByReference.)]
    (when-not (zero? (.pdf_save lib (pdf-ptr p) path code))
      (throw (ex-info "pdf_oxide: save failed" {:code (.getValue code) :op "save"})))))

(defn save-to-bytes ^bytes [^Pdf p]
  (let [len (IntByReference.) code (IntByReference.)
        ptr (.pdf_save_to_bytes lib (pdf-ptr p) len code)]
    (when (nil? ptr) (throw (ex-info "pdf_oxide: save-to-bytes failed" {:code (.getValue code)})))
    (let [n (max 0 (.getValue len)) out (.getByteArray ptr 0 n)]
      (.free_string lib ptr) out)))
