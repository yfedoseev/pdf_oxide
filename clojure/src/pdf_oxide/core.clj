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
  (:import [com.sun.jna NativeLibrary Pointer Function Memory]
           [com.sun.jna.ptr IntByReference ByteByReference FloatByReference DoubleByReference]
           [java.io Closeable]))

(def ^NativeLibrary ^:private nlib (NativeLibrary/getInstance "pdf_oxide"))

(defn- f ^Function [name] (.getFunction nlib name))

;; Typed invocation helpers (Function.invoke dispatches on the return Class).
(defn- ->ptr  ^Pointer [name args] (.invoke (f name) Pointer (object-array args)))
(defn- ->int  ^long    [name args] (.invokeInt (f name) (object-array args)))
(defn- ->bool ^Boolean [name args] (.invoke (f name) Boolean (object-array args)))
(defn- ->float ^double [name args] (.invokeFloat (f name) (object-array args)))
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

(defn to-html-all [^Document d]
  (let [code (IntByReference.)]
    (take-string (->ptr "pdf_document_to_html_all" [(doc-ptr d) code]) (.getValue code) "to-html-all")))

(defn to-plain-text-all [^Document d]
  (let [code (IntByReference.)]
    (take-string (->ptr "pdf_document_to_plain_text_all" [(doc-ptr d) code]) (.getValue code) "to-plain-text-all")))

(defn authenticate
  "Authenticate against a password-protected Document. Returns true on success,
   false for a wrong password (no error)."
  [^Document d password]
  (let [code (IntByReference.)]
    (->bool "pdf_document_authenticate" [(doc-ptr d) password code])))

;; ── Phase-1 element extraction ──────────────────────────────────────────────────
;; Each extractor calls the C-ABI list entry point (NULL → throw), reads every
;; element into Clojure maps, then frees the list once. Bbox is a {:x :y :width
;; :height} map; owned C strings are copied + freed via take-string.
(defn- get-bbox
  "Read a list element's bbox via the C out-param accessor → {:x :y :width :height}."
  [cname ^Pointer lst ^long index]
  (let [x (FloatByReference.) y (FloatByReference.)
        w (FloatByReference.) h (FloatByReference.)
        code (IntByReference.)]
    (->void cname [lst (int index) x y w h code])
    {:x (.getValue x) :y (.getValue y) :width (.getValue w) :height (.getValue h)}))

(defn extract-chars
  "Extract individual characters from a 0-based page.
   Returns a vector of {:character :bbox :font-name :font-size}."
  [^Document d page]
  (let [code (IntByReference.)
        lst (->ptr "pdf_document_extract_chars" [(doc-ptr d) (int page) code])]
    (when (nil? lst) (throw (ex-info "pdf_oxide: extract-chars failed" {:code (.getValue code) :op "extract-chars"})))
    (try
      (let [n (->int "pdf_oxide_char_count" [lst])]
        (mapv (fn [i]
                (let [c (IntByReference.) fs (IntByReference.)]
                  {:character (bit-and (->int "pdf_oxide_char_get_char" [lst (int i) c]) 0xffffffff)
                   :bbox (get-bbox "pdf_oxide_char_get_bbox" lst i)
                   :font-name (take-string (->ptr "pdf_oxide_char_get_font_name" [lst (int i) c]) (.getValue c) "extract-chars")
                   :font-size (->float "pdf_oxide_char_get_font_size" [lst (int i) fs])}))
              (range n)))
      (finally (->void "pdf_oxide_char_list_free" [lst])))))

(defn extract-words
  "Extract words from a 0-based page.
   Returns a vector of {:text :bbox :font-name :font-size :bold}."
  [^Document d page]
  (let [code (IntByReference.)
        lst (->ptr "pdf_document_extract_words" [(doc-ptr d) (int page) code])]
    (when (nil? lst) (throw (ex-info "pdf_oxide: extract-words failed" {:code (.getValue code) :op "extract-words"})))
    (try
      (let [n (->int "pdf_oxide_word_count" [lst])]
        (mapv (fn [i]
                (let [c (IntByReference.) fs (IntByReference.) bc (IntByReference.)]
                  {:text (take-string (->ptr "pdf_oxide_word_get_text" [lst (int i) c]) (.getValue c) "extract-words")
                   :bbox (get-bbox "pdf_oxide_word_get_bbox" lst i)
                   :font-name (take-string (->ptr "pdf_oxide_word_get_font_name" [lst (int i) c]) (.getValue c) "extract-words")
                   :font-size (->float "pdf_oxide_word_get_font_size" [lst (int i) fs])
                   :bold (->bool "pdf_oxide_word_is_bold" [lst (int i) bc])}))
              (range n)))
      (finally (->void "pdf_oxide_word_list_free" [lst])))))

(defn extract-text-lines
  "Extract text lines from a 0-based page.
   Returns a vector of {:text :bbox :word-count}."
  [^Document d page]
  (let [code (IntByReference.)
        lst (->ptr "pdf_document_extract_text_lines" [(doc-ptr d) (int page) code])]
    (when (nil? lst) (throw (ex-info "pdf_oxide: extract-text-lines failed" {:code (.getValue code) :op "extract-text-lines"})))
    (try
      (let [n (->int "pdf_oxide_line_count" [lst])]
        (mapv (fn [i]
                (let [c (IntByReference.) wc (IntByReference.)]
                  {:text (take-string (->ptr "pdf_oxide_line_get_text" [lst (int i) c]) (.getValue c) "extract-text-lines")
                   :bbox (get-bbox "pdf_oxide_line_get_bbox" lst i)
                   :word-count (->int "pdf_oxide_line_get_word_count" [lst (int i) wc])}))
              (range n)))
      (finally (->void "pdf_oxide_line_list_free" [lst])))))

(defn extract-tables
  "Extract tables from a 0-based page. Returns a vector of
   {:row-count :col-count :has-header :cell}, where :cell is a fn (cell row col)
   returning the cell text string."
  [^Document d page]
  (let [code (IntByReference.)
        lst (->ptr "pdf_document_extract_tables" [(doc-ptr d) (int page) code])]
    (when (nil? lst) (throw (ex-info "pdf_oxide: extract-tables failed" {:code (.getValue code) :op "extract-tables"})))
    (try
      (let [n (->int "pdf_oxide_table_count" [lst])]
        (mapv (fn [i]
                (let [rc (IntByReference.) cc (IntByReference.) hc (IntByReference.)]
                  {:row-count (->int "pdf_oxide_table_get_row_count" [lst (int i) rc])
                   :col-count (->int "pdf_oxide_table_get_col_count" [lst (int i) cc])
                   :has-header (->bool "pdf_oxide_table_has_header" [lst (int i) hc])
                   :cell (fn [row col]
                           (let [tc (IntByReference.)]
                             (take-string (->ptr "pdf_oxide_table_get_cell_text" [lst (int i) (int row) (int col) tc])
                                          (.getValue tc) "extract-tables")))}))
              (range n)))
      (finally (->void "pdf_oxide_table_list_free" [lst])))))

;; ── Phase-2 element extraction ──────────────────────────────────────────────────
;; Same shape as Phase-1: call the C-ABI list entry point (NULL → throw), read
;; every element into Clojure maps, then free the list once. Owned C strings are
;; copied + freed via take-string; bbox via the shared get-bbox out-param helper.
(defn embedded-fonts
  "Extract embedded fonts from a 0-based page.
   Returns a vector of {:name :type :encoding :embedded :subset}."
  [^Document d page]
  (let [code (IntByReference.)
        lst (->ptr "pdf_document_get_embedded_fonts" [(doc-ptr d) (int page) code])]
    (when (nil? lst) (throw (ex-info "pdf_oxide: embedded-fonts failed" {:code (.getValue code) :op "embedded-fonts"})))
    (try
      (let [n (->int "pdf_oxide_font_count" [lst])]
        (mapv (fn [i]
                (let [c (IntByReference.)]
                  {:name (take-string (->ptr "pdf_oxide_font_get_name" [lst (int i) c]) (.getValue c) "embedded-fonts")
                   :type (take-string (->ptr "pdf_oxide_font_get_type" [lst (int i) c]) (.getValue c) "embedded-fonts")
                   :encoding (take-string (->ptr "pdf_oxide_font_get_encoding" [lst (int i) c]) (.getValue c) "embedded-fonts")
                   :embedded (->bool "pdf_oxide_font_is_embedded" [lst (int i) c])
                   :subset (->bool "pdf_oxide_font_is_subset" [lst (int i) c])}))
              (range n)))
      (finally (->void "pdf_oxide_font_list_free" [lst])))))

(defn embedded-images
  "Extract embedded images from a 0-based page.
   Returns a vector of {:width :height :bits-per-component :format :colorspace :data}."
  [^Document d page]
  (let [code (IntByReference.)
        lst (->ptr "pdf_document_get_embedded_images" [(doc-ptr d) (int page) code])]
    (when (nil? lst) (throw (ex-info "pdf_oxide: embedded-images failed" {:code (.getValue code) :op "embedded-images"})))
    (try
      (let [n (->int "pdf_oxide_image_count" [lst])]
        (mapv (fn [i]
                (let [c (IntByReference.) len (IntByReference.)
                      ptr (->ptr "pdf_oxide_image_get_data" [lst (int i) len c])
                      sz (max 0 (.getValue len))
                      data (when ptr (let [b (.getByteArray ptr 0 sz)] (free-bytes ptr) b))]
                  {:width (->int "pdf_oxide_image_get_width" [lst (int i) c])
                   :height (->int "pdf_oxide_image_get_height" [lst (int i) c])
                   :bits-per-component (->int "pdf_oxide_image_get_bits_per_component" [lst (int i) c])
                   :format (take-string (->ptr "pdf_oxide_image_get_format" [lst (int i) c]) (.getValue c) "embedded-images")
                   :colorspace (take-string (->ptr "pdf_oxide_image_get_colorspace" [lst (int i) c]) (.getValue c) "embedded-images")
                   :data (or data (byte-array 0))}))
              (range n)))
      (finally (->void "pdf_oxide_image_list_free" [lst])))))

(defn page-annotations
  "Extract annotations from a 0-based page.
   Returns a vector of {:type :subtype :content :author :rect :border-width}."
  [^Document d page]
  (let [code (IntByReference.)
        lst (->ptr "pdf_document_get_page_annotations" [(doc-ptr d) (int page) code])]
    (when (nil? lst) (throw (ex-info "pdf_oxide: page-annotations failed" {:code (.getValue code) :op "page-annotations"})))
    (try
      (let [n (->int "pdf_oxide_annotation_count" [lst])]
        (mapv (fn [i]
                (let [c (IntByReference.)]
                  {:type (take-string (->ptr "pdf_oxide_annotation_get_type" [lst (int i) c]) (.getValue c) "page-annotations")
                   :subtype (take-string (->ptr "pdf_oxide_annotation_get_subtype" [lst (int i) c]) (.getValue c) "page-annotations")
                   :content (take-string (->ptr "pdf_oxide_annotation_get_content" [lst (int i) c]) (.getValue c) "page-annotations")
                   :author (take-string (->ptr "pdf_oxide_annotation_get_author" [lst (int i) c]) (.getValue c) "page-annotations")
                   :rect (get-bbox "pdf_oxide_annotation_get_rect" lst i)
                   :border-width (->float "pdf_oxide_annotation_get_border_width" [lst (int i) c])}))
              (range n)))
      (finally (->void "pdf_oxide_annotation_list_free" [lst])))))

(defn extract-paths
  "Extract vector paths from a 0-based page.
   Returns a vector of {:bbox :stroke-width :has-stroke :has-fill :operation-count}."
  [^Document d page]
  (let [code (IntByReference.)
        lst (->ptr "pdf_document_extract_paths" [(doc-ptr d) (int page) code])]
    (when (nil? lst) (throw (ex-info "pdf_oxide: extract-paths failed" {:code (.getValue code) :op "extract-paths"})))
    (try
      (let [n (->int "pdf_oxide_path_count" [lst])]
        (mapv (fn [i]
                (let [c (IntByReference.)]
                  {:bbox (get-bbox "pdf_oxide_path_get_bbox" lst i)
                   :stroke-width (->float "pdf_oxide_path_get_stroke_width" [lst (int i) c])
                   :has-stroke (->bool "pdf_oxide_path_has_stroke" [lst (int i) c])
                   :has-fill (->bool "pdf_oxide_path_has_fill" [lst (int i) c])
                   :operation-count (->int "pdf_oxide_path_get_operation_count" [lst (int i) c])}))
              (range n)))
      (finally (->void "pdf_oxide_path_list_free" [lst])))))

(defn- read-search-results
  "Read a FfiSearchResults handle into a vector of {:text :page :bbox}, then free
   it via pdf_oxide_search_result_free (NOT _list_free)."
  [^Pointer lst op]
  (try
    (let [n (->int "pdf_oxide_search_result_count" [lst])]
      (mapv (fn [i]
              (let [c (IntByReference.)]
                {:text (take-string (->ptr "pdf_oxide_search_result_get_text" [lst (int i) c]) (.getValue c) op)
                 :page (->int "pdf_oxide_search_result_get_page" [lst (int i) c])
                 :bbox (get-bbox "pdf_oxide_search_result_get_bbox" lst i)}))
            (range n)))
    (finally (->void "pdf_oxide_search_result_free" [lst]))))

(defn search
  "Search a 0-based page for `term`. Returns a vector of {:text :page :bbox}."
  [^Document d page term case-sensitive]
  (let [code (IntByReference.)
        lst (->ptr "pdf_document_search_page" [(doc-ptr d) (int page) term (boolean case-sensitive) code])]
    (when (nil? lst) (throw (ex-info "pdf_oxide: search failed" {:code (.getValue code) :op "search"})))
    (read-search-results lst "search")))

(defn search-all
  "Search the whole document for `term`. Returns a vector of {:text :page :bbox}."
  [^Document d term case-sensitive]
  (let [code (IntByReference.)
        lst (->ptr "pdf_document_search_all" [(doc-ptr d) term (boolean case-sensitive) code])]
    (when (nil? lst) (throw (ex-info "pdf_oxide: search-all failed" {:code (.getValue code) :op "search-all"})))
    (read-search-results lst "search-all")))

;; ── Page ────────────────────────────────────────────────────────────────────
;; Holds a strong reference to its Document (keeps it alive) plus a 0-based index;
;; methods delegate to the existing per-page Document fns.
(deftype Page [^Document doc ^long index])

(defn page
  "Return a Page view over a 0-based page index of the Document."
  [^Document d index]
  (Page. d (long index)))

(defn page-text [^Page pg] (extract-text (.-doc pg) (.-index pg)))
(defn page-markdown [^Page pg] (to-markdown (.-doc pg) (.-index pg)))
(defn page-html [^Page pg] (to-html (.-doc pg) (.-index pg)))
(defn page-plain-text [^Page pg] (to-plain-text (.-doc pg) (.-index pg)))

;; ── Phase-3 page rendering ────────────────────────────────────────────────────
;; A RenderedImage owns the native FfiRenderedImage handle (freed on close via
;; pdf_rendered_image_free, so use with `with-open`). width/height/data are read
;; eagerly into Clojure on construction; data bytes are copied out of the C buffer
;; and freed via free_bytes. `save` uses the still-live handle.
(deftype RenderedImage [state ^long width ^long height ^bytes data]   ; state = (atom Pointer-or-nil)
  Closeable
  (close [_] (when-let [h @state] (->void "pdf_rendered_image_free" [h]) (reset! state nil))))

(defn rendered-image-width  ^long [^RenderedImage img] (.-width img))
(defn rendered-image-height ^long [^RenderedImage img] (.-height img))
(defn rendered-image-data  ^bytes [^RenderedImage img] (.-data img))

(defn- img-ptr ^Pointer [^RenderedImage img]
  (or @(.-state img) (throw (ex-info "RenderedImage is closed" {}))))

(defn- wrap-rendered-image
  "Read width/height/data eagerly off an FfiRenderedImage handle (copying + freeing
   the data buffer via free_bytes) and return a RenderedImage owning the handle."
  [^Pointer h op]
  (let [code (IntByReference.)
        w (->int "pdf_get_rendered_image_width" [h code])
        ht (->int "pdf_get_rendered_image_height" [h code])
        len (IntByReference.)
        ptr (->ptr "pdf_get_rendered_image_data" [h len code])
        sz (max 0 (.getValue len))
        data (if ptr (let [b (.getByteArray ptr 0 sz)] (free-bytes ptr) b) (byte-array 0))]
    (when (nil? ptr)
      (->void "pdf_rendered_image_free" [h])
      (throw (ex-info (str "pdf_oxide: " op " (image data) failed") {:code (.getValue code) :op op})))
    (RenderedImage. (atom h) w ht data)))

(defn rendered-image-save
  "Save a RenderedImage to `path` (encoded by its format). Throws on error."
  [^RenderedImage img path]
  (let [code (IntByReference.)]
    (when-not (zero? (->int "pdf_save_rendered_image" [(img-ptr img) path code]))
      (throw (ex-info "pdf_oxide: rendered-image-save failed" {:code (.getValue code) :op "rendered-image-save"})))))

(defn render-page
  "Render a 0-based page to a RenderedImage. `format` is an int image format
   (0 = PNG, default). Returns a Closeable RenderedImage."
  ([^Document d page] (render-page d page 0))
  ([^Document d page format]
   (let [code (IntByReference.)
         h (->ptr "pdf_render_page" [(doc-ptr d) (int page) (int format) code])]
     (when (nil? h) (throw (ex-info "pdf_oxide: render-page failed" {:code (.getValue code) :op "render-page"})))
     (wrap-rendered-image h "render-page"))))

(defn render-page-zoom
  "Render a 0-based page at `zoom` (float scale) to a RenderedImage.
   `format` is an int image format (0 = PNG, default)."
  ([^Document d page zoom] (render-page-zoom d page zoom 0))
  ([^Document d page zoom format]
   (let [code (IntByReference.)
         h (->ptr "pdf_render_page_zoom" [(doc-ptr d) (int page) (float zoom) (int format) code])]
     (when (nil? h) (throw (ex-info "pdf_oxide: render-page-zoom failed" {:code (.getValue code) :op "render-page-zoom"})))
     (wrap-rendered-image h "render-page-zoom"))))

(defn render-page-thumbnail
  "Render a 0-based page fit to `size` pixels to a RenderedImage.
   `format` is an int image format (0 = PNG, default)."
  ([^Document d page size] (render-page-thumbnail d page size 0))
  ([^Document d page size format]
   (let [code (IntByReference.)
         h (->ptr "pdf_render_page_thumbnail" [(doc-ptr d) (int page) (int size) (int format) code])]
     (when (nil? h) (throw (ex-info "pdf_oxide: render-page-thumbnail failed" {:code (.getValue code) :op "render-page-thumbnail"})))
     (wrap-rendered-image h "render-page-thumbnail"))))

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

;; ── DocumentEditor ──────────────────────────────────────────────────────────────
;; Mutating editor handle over the document_editor_* C ABI; mirrors the Document /
;; Pdf handle pattern. Owns a native DocumentEditor* freed on close (Closeable, so
;; use with `with-open`). Status-returning C fns yield 0 on success — any non-zero
;; status OR a non-zero error_code is treated as a failure and throws ex-info
;; {:code …}. is_* queries return int32 exposed as bool (1 = true). Page indices
;; are 0-based; box getters reuse the {:x :y :width :height} bbox map shape.
(deftype DocumentEditor [state]   ; state = (atom Pointer-or-nil)
  Closeable
  (close [_] (when-let [h @state] (->void "document_editor_free" [h]) (reset! state nil))))

(defn- ed-ptr ^Pointer [^DocumentEditor e]
  (or @(.-state e) (throw (ex-info "DocumentEditor is closed" {}))))

(defn- ed-check
  "Throw if a status/error_code pair signals failure (status non-zero OR code non-zero)."
  [^long status ^IntByReference code op]
  (when (or (not (zero? status)) (not (zero? (.getValue code))))
    (throw (ex-info (str "pdf_oxide: " op " failed") {:code (.getValue code) :status status :op op})))
  status)

(defn- ed-status
  "Invoke a status-returning document_editor_* fn (trailing error_code out-param) and
   throw on failure. `args` excludes the trailing error_code."
  [cname args op]
  (let [code (IntByReference.)
        status (->int cname (conj (vec args) code))]
    (ed-check status code op)))

(defn- ed-string
  "Invoke an owned-char*-returning document_editor_* fn (trailing error_code) and
   return the copied string (freed via free_string)."
  [cname args op]
  (let [code (IntByReference.)]
    (take-string (->ptr cname (conj (vec args) code)) (.getValue code) op)))

(defn- ed-bytes
  "Invoke an owned-uint8*-returning document_editor_* fn whose trailing out-params are
   (out_len, error_code). Returns the copied byte array (freed via free_bytes)."
  ^bytes [cname args op]
  (let [len (IntByReference.) code (IntByReference.)
        ptr (->ptr cname (conj (vec args) len code))]
    (when (nil? ptr) (throw (ex-info (str "pdf_oxide: " op " failed") {:code (.getValue code) :op op})))
    (let [n (max 0 (.getValue len)) out (.getByteArray ptr 0 n)]
      (free-bytes ptr) out)))

(defn- ed-box
  "Read a page box via a double x,y,w,h out-param getter → {:x :y :width :height}."
  [cname ^DocumentEditor e ^long page op]
  (let [x (DoubleByReference.) y (DoubleByReference.)
        w (DoubleByReference.) h (DoubleByReference.)
        code (IntByReference.)
        status (->int cname [(ed-ptr e) page x y w h code])]
    (ed-check status code op)
    {:x (.getValue x) :y (.getValue y) :width (.getValue w) :height (.getValue h)}))

(defn- ed-bool
  "Invoke an int32 is_* query (1=true, 0=false, -1=error). No error_code out-param."
  [cname args op]
  (let [r (->int cname (vec args))]
    (when (neg? r) (throw (ex-info (str "pdf_oxide: " op " failed") {:status r :op op})))
    (= r 1)))

;; ── open / lifecycle ────────────────────────────────────────────────────────────
(defn open-editor
  "Open a PDF for editing from a path. Returns a Closeable DocumentEditor."
  [path]
  (let [code (IntByReference.)
        h (->ptr "document_editor_open" [path code])]
    (when (nil? h) (throw (ex-info "pdf_oxide: open-editor failed" {:code (.getValue code) :op "open-editor"})))
    (DocumentEditor. (atom h))))

(defn open-editor-from-bytes
  "Open a DocumentEditor from a byte array. Returns a Closeable DocumentEditor."
  [^bytes data]
  (let [code (IntByReference.)
        h (->ptr "document_editor_open_from_bytes" [data (long (alength data)) code])]
    (when (nil? h) (throw (ex-info "pdf_oxide: open-editor-from-bytes failed" {:code (.getValue code) :op "open-editor-from-bytes"})))
    (DocumentEditor. (atom h))))

(defn editor-modified? [^DocumentEditor e]
  (->bool "document_editor_is_modified" [(ed-ptr e)]))

(defn editor-page-count [^DocumentEditor e]
  (let [code (IntByReference.) n (->int "document_editor_get_page_count" [(ed-ptr e) code])]
    (when (neg? n) (throw (ex-info "pdf_oxide: editor-page-count failed" {:code (.getValue code) :op "editor-page-count"})))
    n))

(defn editor-version
  "PDF version as a map {:major _ :minor _}."
  [^DocumentEditor e]
  (let [maj (ByteByReference.) min (ByteByReference.)]
    (->void "document_editor_get_version" [(ed-ptr e) maj min])
    {:major (bit-and (.getValue maj) 0xff) :minor (bit-and (.getValue min) 0xff)}))

(defn editor-source-path [^DocumentEditor e]
  (ed-string "document_editor_get_source_path" [(ed-ptr e)] "editor-source-path"))

;; ── metadata ──────────────────────────────────────────────────────────────────
(defn editor-producer [^DocumentEditor e]
  (ed-string "document_editor_get_producer" [(ed-ptr e)] "editor-producer"))

(defn set-editor-producer [^DocumentEditor e value]
  (ed-status "document_editor_set_producer" [(ed-ptr e) value] "set-editor-producer"))

(defn editor-creation-date [^DocumentEditor e]
  (ed-string "document_editor_get_creation_date" [(ed-ptr e)] "editor-creation-date"))

(defn set-editor-creation-date [^DocumentEditor e date-str]
  (ed-status "document_editor_set_creation_date" [(ed-ptr e) date-str] "set-editor-creation-date"))

;; ── page operations ─────────────────────────────────────────────────────────────
(defn editor-delete-page [^DocumentEditor e page]
  (ed-status "document_editor_delete_page" [(ed-ptr e) (int page)] "editor-delete-page"))

(defn editor-move-page [^DocumentEditor e from to]
  (ed-status "document_editor_move_page" [(ed-ptr e) (int from) (int to)] "editor-move-page"))

(defn editor-rotate-page-by [^DocumentEditor e page degrees]
  (ed-status "document_editor_rotate_page_by" [(ed-ptr e) (long page) (int degrees)] "editor-rotate-page-by"))

(defn editor-rotate-all-pages [^DocumentEditor e degrees]
  (ed-status "document_editor_rotate_all_pages" [(ed-ptr e) (int degrees)] "editor-rotate-all-pages"))

(defn set-editor-page-rotation [^DocumentEditor e page degrees]
  (ed-status "document_editor_set_page_rotation" [(ed-ptr e) (int page) (int degrees)] "set-editor-page-rotation"))

(defn editor-page-rotation
  "Page rotation in degrees (0/90/180/270)."
  [^DocumentEditor e page]
  (let [code (IntByReference.)
        r (->int "document_editor_get_page_rotation" [(ed-ptr e) (int page) code])]
    (when (neg? r) (throw (ex-info "pdf_oxide: editor-page-rotation failed" {:code (.getValue code) :op "editor-page-rotation"})))
    r))

(defn editor-crop-margins [^DocumentEditor e left right top bottom]
  (ed-status "document_editor_crop_margins"
             [(ed-ptr e) (float left) (float right) (float top) (float bottom)] "editor-crop-margins"))

;; ── boxes ───────────────────────────────────────────────────────────────────────
(defn editor-page-crop-box
  "CropBox of a page as {:x :y :width :height} (0,0,0,0 if unset)."
  [^DocumentEditor e page]
  (ed-box "document_editor_get_page_crop_box" e page "editor-page-crop-box"))

(defn set-editor-page-crop-box [^DocumentEditor e page x y w h]
  (ed-status "document_editor_set_page_crop_box"
             [(ed-ptr e) (long page) (double x) (double y) (double w) (double h)] "set-editor-page-crop-box"))

(defn editor-page-media-box
  "MediaBox of a page as {:x :y :width :height}."
  [^DocumentEditor e page]
  (ed-box "document_editor_get_page_media_box" e page "editor-page-media-box"))

(defn set-editor-page-media-box [^DocumentEditor e page x y w h]
  (ed-status "document_editor_set_page_media_box"
             [(ed-ptr e) (long page) (double x) (double y) (double w) (double h)] "set-editor-page-media-box"))

;; ── redaction ─────────────────────────────────────────────────────────────────
(defn editor-apply-all-redactions [^DocumentEditor e]
  (ed-status "document_editor_apply_all_redactions" [(ed-ptr e)] "editor-apply-all-redactions"))

(defn editor-apply-page-redactions [^DocumentEditor e page]
  (ed-status "document_editor_apply_page_redactions" [(ed-ptr e) (long page)] "editor-apply-page-redactions"))

(defn editor-erase-region [^DocumentEditor e page x y w h]
  (ed-status "document_editor_erase_region"
             [(ed-ptr e) (int page) (float x) (float y) (float w) (float h)] "editor-erase-region"))

(defn editor-erase-regions
  "Erase multiple rectangles on `page`. `rects` is a seq of [x y w h] quads (doubles)."
  [^DocumentEditor e page rects]
  (let [flat (mapcat (fn [[x y w h]] [(double x) (double y) (double w) (double h)]) rects)
        n (count rects)
        mem (when (pos? n)
              (let [m (Memory. (* 8 (count flat)))]
                (doseq [[i v] (map-indexed vector flat)] (.setDouble m (* 8 (long i)) (double v)))
                m))]
    (ed-status "document_editor_erase_regions" [(ed-ptr e) (long page) mem (long n)] "editor-erase-regions")))

(defn editor-clear-erase-regions [^DocumentEditor e page]
  (ed-status "document_editor_clear_erase_regions" [(ed-ptr e) (long page)] "editor-clear-erase-regions"))

(defn editor-page-marked-for-redaction? [^DocumentEditor e page]
  (ed-bool "document_editor_is_page_marked_for_redaction" [(ed-ptr e) (long page)] "editor-page-marked-for-redaction?"))

(defn editor-unmark-page-for-redaction [^DocumentEditor e page]
  (ed-status "document_editor_unmark_page_for_redaction" [(ed-ptr e) (long page)] "editor-unmark-page-for-redaction"))

;; ── flatten ───────────────────────────────────────────────────────────────────
(defn editor-flatten-forms [^DocumentEditor e]
  (ed-status "document_editor_flatten_forms" [(ed-ptr e)] "editor-flatten-forms"))

(defn editor-flatten-forms-on-page [^DocumentEditor e page]
  (ed-status "document_editor_flatten_forms_on_page" [(ed-ptr e) (int page)] "editor-flatten-forms-on-page"))

(defn editor-flatten-annotations [^DocumentEditor e page]
  (ed-status "document_editor_flatten_annotations" [(ed-ptr e) (int page)] "editor-flatten-annotations"))

(defn editor-flatten-all-annotations [^DocumentEditor e]
  (ed-status "document_editor_flatten_all_annotations" [(ed-ptr e)] "editor-flatten-all-annotations"))

(defn editor-flatten-warnings-count
  "Number of warnings from the last form-flattening save."
  [^DocumentEditor e]
  (let [r (->int "document_editor_flatten_warnings_count" [(ed-ptr e)])]
    (when (neg? r) (throw (ex-info "pdf_oxide: editor-flatten-warnings-count failed" {:status r :op "editor-flatten-warnings-count"})))
    r))

(defn editor-flatten-warning [^DocumentEditor e index]
  (ed-string "document_editor_flatten_warning" [(ed-ptr e) (int index)] "editor-flatten-warning"))

(defn editor-page-marked-for-flatten? [^DocumentEditor e page]
  (ed-bool "document_editor_is_page_marked_for_flatten" [(ed-ptr e) (long page)] "editor-page-marked-for-flatten?"))

(defn editor-unmark-page-for-flatten [^DocumentEditor e page]
  (ed-status "document_editor_unmark_page_for_flatten" [(ed-ptr e) (long page)] "editor-unmark-page-for-flatten"))

;; ── forms / merge / convert / embed ─────────────────────────────────────────────
(defn set-editor-form-field-value [^DocumentEditor e name value]
  (ed-status "document_editor_set_form_field_value" [(ed-ptr e) name value] "set-editor-form-field-value"))

(defn editor-merge-from [^DocumentEditor e source-path]
  (ed-status "document_editor_merge_from" [(ed-ptr e) source-path] "editor-merge-from"))

(defn editor-merge-from-bytes [^DocumentEditor e ^bytes data]
  (ed-status "document_editor_merge_from_bytes" [(ed-ptr e) data (long (alength data))] "editor-merge-from-bytes"))

(defn editor-convert-to-pdf-a
  "Convert the document to PDF/A in-place. `level`: 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u."
  [^DocumentEditor e level]
  (ed-status "document_editor_convert_to_pdf_a" [(ed-ptr e) (int level)] "editor-convert-to-pdf-a"))

(defn editor-embed-file [^DocumentEditor e name ^bytes data]
  (ed-status "document_editor_embed_file" [(ed-ptr e) name data (long (alength data))] "editor-embed-file"))

(defn editor-extract-pages-to-bytes
  "Extract a subset of 0-based `pages` (a seq of ints) into a new in-memory PDF (bytes)."
  ^bytes [^DocumentEditor e pages]
  (let [n (count pages)
        mem (when (pos? n)
              (let [m (Memory. (* 4 n))]
                (doseq [[i v] (map-indexed vector pages)] (.setInt m (* 4 (long i)) (int v)))
                m))]
    (ed-bytes "document_editor_extract_pages_to_bytes" [(ed-ptr e) mem (long n)] "editor-extract-pages-to-bytes")))

;; ── save ────────────────────────────────────────────────────────────────────────
(defn editor-save [^DocumentEditor e path]
  (ed-status "document_editor_save" [(ed-ptr e) path] "editor-save"))

(defn editor-save-to-bytes ^bytes [^DocumentEditor e]
  (ed-bytes "document_editor_save_to_bytes" [(ed-ptr e)] "editor-save-to-bytes"))

(defn editor-save-encrypted [^DocumentEditor e path user-password owner-password]
  (ed-status "document_editor_save_encrypted" [(ed-ptr e) path user-password owner-password] "editor-save-encrypted"))

(defn editor-save-encrypted-to-bytes ^bytes [^DocumentEditor e user-password owner-password]
  (ed-bytes "document_editor_save_encrypted_to_bytes" [(ed-ptr e) user-password owner-password] "editor-save-encrypted-to-bytes"))

(defn editor-save-to-bytes-with-options
  "Save with compression / GC / linearize options. Returns the PDF bytes."
  ^bytes [^DocumentEditor e compress garbage-collect linearize]
  (ed-bytes "document_editor_save_to_bytes_with_options"
            [(ed-ptr e) (boolean compress) (boolean garbage-collect) (boolean linearize)]
            "editor-save-to-bytes-with-options"))
