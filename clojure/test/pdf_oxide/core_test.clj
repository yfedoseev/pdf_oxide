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

(deftest error-path
  (is (thrown? clojure.lang.ExceptionInfo (pdf/open "/nonexistent/nope.pdf"))))
