;; One test per public fn — mirrors the api_coverage convention used by every
;; pdf_oxide binding. Self-contained: builds its own PDF from Markdown.
(ns pdf-oxide.core-test
  (:require [clojure.test :refer [deftest is testing]]
            [pdf-oxide.core :as pdf])
  (:import [java.io File]))

(defn sample-pdf ^bytes []
  (with-open [p (pdf/from-markdown "# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n")]
    (pdf/to-bytes p)))

(deftest pdf-builder
  (testing "from-markdown + to-bytes"
    (with-open [p (pdf/from-markdown "# md\n\nbody\n")] (is (> (alength (pdf/to-bytes p)) 100))))
  (testing "from-html"
    (with-open [p (pdf/from-html "<h1>h</h1><p>b</p>")] (is (> (alength (pdf/to-bytes p)) 100))))
  (testing "from-text"
    (with-open [p (pdf/from-text "plain text body")] (is (> (alength (pdf/to-bytes p)) 100))))
  (testing "save"
    (let [f (File/createTempFile "pdfoxide-clj" ".pdf")]
      (with-open [p (pdf/from-markdown "# f\n\nx\n")] (pdf/save p (.getAbsolutePath f)))
      (is (> (.length f) 100)) (.delete f))))

(deftest document-open
  (testing "open-from-bytes + page-count"
    (with-open [d (pdf/open-from-bytes (sample-pdf))] (is (>= (pdf/page-count d) 1))))
  (testing "open (path)"
    (let [f (File/createTempFile "pdfoxide-clj-open" ".pdf")]
      (with-open [p (pdf/from-markdown "# f\n\nx\n")] (pdf/save p (.getAbsolutePath f)))
      (with-open [d (pdf/open (.getAbsolutePath f))] (is (>= (pdf/page-count d) 1)))
      (.delete f))))

(deftest document-inspection-extraction
  (with-open [d (pdf/open-from-bytes (sample-pdf))]
    (is (>= (:major (pdf/version d)) 1))           ; version
    (is (false? (pdf/encrypted? d)))               ; encrypted?
    (pdf/structure-tree? d)                         ; structure-tree? (smoke)
    (is (re-find #"Alpha" (pdf/extract-text d 0)))  ; extract-text
    (is (seq (pdf/to-plain-text d 0)))              ; to-plain-text
    (is (seq (pdf/to-markdown d 0)))                ; to-markdown
    (is (re-find #"<" (pdf/to-html d 0)))           ; to-html
    (is (seq (pdf/to-markdown-all d)))              ; to-markdown-all
    (is (re-find #"<" (pdf/to-html-all d)))         ; to-html-all
    (is (seq (pdf/to-plain-text-all d)))            ; to-plain-text-all
    (is (instance? Boolean (pdf/authenticate d "")) ); authenticate (returns a bool, no error)
    (let [pg (pdf/page d 0)]                         ; page (0-based)
      (is (re-find #"Alpha" (pdf/page-text pg)))     ; page-text
      (is (seq (pdf/page-markdown pg)))              ; page-markdown
      (is (seq (pdf/page-html pg)))                  ; page-html
      (is (seq (pdf/page-plain-text pg))))           ; page-plain-text
    (is (seq (pdf/extract-structured-json d 0)))))  ; extract-structured-json

(deftest phase1-element-extraction
  (with-open [d (pdf/open-from-bytes (sample-pdf))]
    (testing "extract-words (0-based)"
      (let [words (pdf/extract-words d 0)]
        (is (seq words))                                  ; non-empty
        (is (seq (:text (first words))))                  ; word[0].text non-empty
        (is (map? (:bbox (first words))))                 ; word[0] has a bbox
        (is (number? (:x (:bbox (first words)))))
        (is (instance? Boolean (:bold (first words))))))
    (testing "extract-chars (0-based)"
      (let [chars (pdf/extract-chars d 0)]
        (is (seq chars))                                  ; non-empty
        (is (integer? (:character (first chars))))        ; codepoint as int
        (is (map? (:bbox (first chars))))))
    (testing "extract-text-lines (0-based)"
      (let [lines (pdf/extract-text-lines d 0)]
        (is (seq lines))                                  ; non-empty
        (is (seq (:text (first lines))))
        (is (integer? (:word-count (first lines))))))
    (testing "extract-tables (0-based) returns a list without error"
      (let [tables (pdf/extract-tables d 0)]
        (is (sequential? tables))                         ; may be empty
        (doseq [t tables]
          (is (integer? (:row-count t)))
          (is (integer? (:col-count t)))
          (is (instance? Boolean (:has-header t)))
          (is (fn? (:cell t))))))))

(deftest phase2-element-extraction
  (with-open [d (pdf/open-from-bytes (sample-pdf))]
    (testing "embedded-fonts (0-based) returns a list without error"
      (let [fonts (pdf/embedded-fonts d 0)]
        (is (sequential? fonts))                          ; may be empty
        (doseq [ft fonts]
          (is (string? (:name ft)))
          (is (string? (:type ft)))
          (is (string? (:encoding ft)))
          (is (instance? Boolean (:embedded ft)))
          (is (instance? Boolean (:subset ft))))))
    (testing "embedded-images (0-based) returns a list without error"
      (let [images (pdf/embedded-images d 0)]
        (is (sequential? images))                         ; may be empty
        (doseq [im images]
          (is (integer? (:width im)))
          (is (integer? (:height im)))
          (is (integer? (:bits-per-component im)))
          (is (string? (:format im)))
          (is (string? (:colorspace im)))
          (is (bytes? (:data im))))))
    (testing "page-annotations (0-based) returns a list without error"
      (let [anns (pdf/page-annotations d 0)]
        (is (sequential? anns))                           ; may be empty
        (doseq [a anns]
          (is (string? (:type a)))
          (is (string? (:subtype a)))
          (is (string? (:content a)))
          (is (string? (:author a)))
          (is (map? (:rect a)))
          (is (number? (:border-width a))))))
    (testing "extract-paths (0-based) returns a list without error"
      (let [paths (pdf/extract-paths d 0)]
        (is (sequential? paths))                          ; may be empty
        (doseq [p paths]
          (is (map? (:bbox p)))
          (is (number? (:stroke-width p)))
          (is (instance? Boolean (:has-stroke p)))
          (is (instance? Boolean (:has-fill p)))
          (is (integer? (:operation-count p))))))
    (testing "search (0-based)"
      (let [results (pdf/search d 0 "Alpha" false)]
        (is (seq results))                                ; non-empty
        (is (re-find #"Alpha" (:text (first results))))   ; first result contains Alpha
        (is (>= (:page (first results)) 0))               ; page >= 0
        (is (map? (:bbox (first results))))))
    (testing "search-all"
      (let [results (pdf/search-all d "Alpha" false)]
        (is (seq results))                                ; non-empty
        (is (re-find #"Alpha" (:text (first results))))   ; first result contains Alpha
        (is (>= (:page (first results)) 0))))))           ; page >= 0

(deftest phase3-page-rendering
  (with-open [d (pdf/open-from-bytes (sample-pdf))]
    (testing "render-page (0-based, PNG)"
      (with-open [img (pdf/render-page d 0 0)]
        (is (> (pdf/rendered-image-width img) 0))         ; width > 0
        (is (> (pdf/rendered-image-height img) 0))        ; height > 0
        (is (bytes? (pdf/rendered-image-data img)))
        (is (pos? (alength (pdf/rendered-image-data img)))) ; non-empty data
        (testing "rendered-image-save"
          (let [f (File/createTempFile "pdfoxide-clj-render" ".png")]
            (pdf/rendered-image-save img (.getAbsolutePath f))
            (is (> (.length f) 0)) (.delete f)))))
    (testing "render-page (default format = PNG)"
      (with-open [img (pdf/render-page d 0)]
        (is (> (pdf/rendered-image-width img) 0))))
    (testing "render-page-zoom returns without error"
      (with-open [img (pdf/render-page-zoom d 0 2.0)]
        (is (> (pdf/rendered-image-width img) 0))))
    (testing "render-page-thumbnail returns without error"
      (with-open [img (pdf/render-page-thumbnail d 0 64)]
        (is (> (pdf/rendered-image-width img) 0))))))

(deftest error-path
  (is (thrown? clojure.lang.ExceptionInfo (pdf/open "/nonexistent/nope.pdf"))))
