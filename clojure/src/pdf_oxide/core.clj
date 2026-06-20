;; pdf_oxide — idiomatic Clojure bindings over the C ABI via JNA.
;;
;; Uses JNA's NativeLibrary + Function directly (no Library-extending interface),
;; the idiomatic dynamic-Clojure approach. Document/pdf handles are Closeable
;; deftypes wrapping a JNA Pointer (use with `with-open`); returned C strings/
;; buffers are copied into Clojure and freed via free_string; non-success C-ABI
;; error codes throw ex-info with {:code …}.
;;
;; API surface mirrors the other language bindings; coverage is asserted by
;; pdf-oxide.core-test (one test per public fn).
(ns pdf-oxide.core
  (:import [com.sun.jna NativeLibrary Pointer Function]
           [com.sun.jna.ptr IntByReference ByteByReference]
           [java.io Closeable]))

(def ^NativeLibrary ^:private nlib (NativeLibrary/getInstance "pdf_oxide"))

(defn- f ^Function [name] (.getFunction nlib name))

;; Typed invocation helpers (Function.invoke dispatches on the return Class).
(defn- ->ptr  ^Pointer [name args] (.invoke (f name) Pointer (object-array args)))
(defn- ->int  ^long    [name args] (.invokeInt (f name) (object-array args)))
(defn- ->bool ^Boolean [name args] (.invoke (f name) Boolean (object-array args)))
(defn- ->void [name args] (.invokeVoid (f name) (object-array args)))
(defn- free-string [^Pointer p] (.invokeVoid (f "free_string") (object-array [p])))
(defn- free-bytes [^Pointer p] (.invokeVoid (f "free_bytes") (object-array [p])))

(defn- take-string [^Pointer p ^long code op]
  (when (nil? p) (throw (ex-info (str "pdf_oxide: " op " failed") {:code code :op op})))
  (let [s (.getString p 0)] (free-string p) s))

;; ── Document ──────────────────────────────────────────────────────────────────
(deftype Document [state]   ; state = (atom Pointer-or-nil)
  Closeable
  (close [_] (when-let [h @state] (->void "pdf_document_free" [h]) (reset! state nil))))

(defn- doc-ptr ^Pointer [^Document d]
  (or @(.-state d) (throw (ex-info "PdfDocument is closed" {}))))

(defn open
  "Open a PDF from a path."
  [path]
  (let [code (IntByReference.)
        h (->ptr "pdf_document_open" [path code])]
    (when (nil? h) (throw (ex-info "pdf_oxide: open failed" {:code (.getValue code) :op "open"})))
    (Document. (atom h))))

(defn open-with-password
  "Open a password-protected PDF."
  [path password]
  (let [code (IntByReference.)
        h (->ptr "pdf_document_open_with_password" [path password code])]
    (when (nil? h) (throw (ex-info "pdf_oxide: open-with-password failed" {:code (.getValue code) :op "open-with-password"})))
    (Document. (atom h))))

(defn open-from-bytes
  "Open a PDF from a byte array."
  [^bytes data]
  (let [code (IntByReference.)
        h (->ptr "pdf_document_open_from_bytes" [data (long (alength data)) code])]
    (when (nil? h) (throw (ex-info "pdf_oxide: open-from-bytes failed" {:code (.getValue code) :op "open-from-bytes"})))
    (Document. (atom h))))

(defn page-count [^Document d]
  (let [code (IntByReference.) n (->int "pdf_document_get_page_count" [(doc-ptr d) code])]
    (when (neg? n) (throw (ex-info "pdf_oxide: page-count failed" {:code (.getValue code)})))
    n))

(defn version
  "PDF version as a map {:major _ :minor _}."
  [^Document d]
  (let [maj (ByteByReference.) min (ByteByReference.)]
    (->void "pdf_document_get_version" [(doc-ptr d) maj min])
    {:major (bit-and (.getValue maj) 0xff) :minor (bit-and (.getValue min) 0xff)}))

(defn encrypted? [^Document d] (->bool "pdf_document_is_encrypted" [(doc-ptr d)]))
(defn structure-tree? [^Document d] (->bool "pdf_document_has_structure_tree" [(doc-ptr d)]))

(defn- str-page [cname ^Document d ^long page op]
  (let [code (IntByReference.)]
    (take-string (->ptr cname [(doc-ptr d) (int page) code]) (.getValue code) op)))

(defn extract-text [^Document d page] (str-page "pdf_document_extract_text" d page "extract-text"))
(defn to-plain-text [^Document d page] (str-page "pdf_document_to_plain_text" d page "to-plain-text"))
(defn to-markdown [^Document d page] (str-page "pdf_document_to_markdown" d page "to-markdown"))
(defn to-html [^Document d page] (str-page "pdf_document_to_html" d page "to-html"))
(defn extract-structured-json [^Document d page]
  (str-page "pdf_document_extract_structured_to_json" d page "extract-structured-json"))

(defn to-markdown-all [^Document d]
  (let [code (IntByReference.)]
    (take-string (->ptr "pdf_document_to_markdown_all" [(doc-ptr d) code]) (.getValue code) "to-markdown-all")))

;; ── Pdf builder ───────────────────────────────────────────────────────────────
(deftype Pdf [state]   ; state = (atom Pointer-or-nil)
  Closeable
  (close [_] (when-let [h @state] (->void "pdf_free" [h]) (reset! state nil))))

(defn- pdf-ptr ^Pointer [^Pdf p]
  (or @(.-state p) (throw (ex-info "Pdf is closed" {}))))

(defn- build [cname input op]
  (let [code (IntByReference.) h (->ptr cname [input code])]
    (when (nil? h) (throw (ex-info (str "pdf_oxide: " op " failed") {:code (.getValue code) :op op})))
    (Pdf. (atom h))))

(defn from-markdown [md] (build "pdf_from_markdown" md "from-markdown"))
(defn from-html [html] (build "pdf_from_html" html "from-html"))
(defn from-text [text] (build "pdf_from_text" text "from-text"))

(defn save [^Pdf p path]
  (let [code (IntByReference.)]
    (when-not (zero? (->int "pdf_save" [(pdf-ptr p) path code]))
      (throw (ex-info "pdf_oxide: save failed" {:code (.getValue code) :op "save"})))))

(defn to-bytes ^bytes [^Pdf p]
  (let [len (IntByReference.) code (IntByReference.)
        ptr (->ptr "pdf_save_to_bytes" [(pdf-ptr p) len code])]
    (when (nil? ptr) (throw (ex-info "pdf_oxide: to-bytes failed" {:code (.getValue code)})))
    (let [n (max 0 (.getValue len)) out (.getByteArray ptr 0 n)]
      (free-bytes ptr) out)))
