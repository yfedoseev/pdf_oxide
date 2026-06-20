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
           [com.sun.jna.ptr IntByReference ByteByReference FloatByReference]
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
    (RenderedImage. (atom h) (long w) (long ht) data)))

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
