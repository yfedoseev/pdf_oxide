;; Coverage for the Clojure facade over the Java binding. Self-contained: builds
;; its own PDF from Markdown, and exercises every public fn in pdf-oxide.core at
;; least once — matching the "one test per public member" standard of the other
;; pdf_oxide bindings. Handles are AutoCloseable -> `with-open`.
(ns pdf-oxide.core-test
  (:require [clojure.test :refer [deftest is]]
            [pdf-oxide.core :as pdf])
  (:import [fyi.oxide.pdf.geometry BBox]))

(defn sample-pdf ^bytes []
  (with-open [p (pdf/from-markdown "# Alpha Heading\n\nHello world from the Clojure facade. Beta gamma.\n")]
    (pdf/save p)))

(deftest creation-from-markdown-and-html
  (let [md (sample-pdf)]
    (is (> (count md) 100))
    (is (= (byte \%) (aget md 0))))
  (with-open [p (pdf/from-html "<h1>Hi</h1><p>body</p>")]
    (is (> (count (pdf/save p)) 100))))

(deftest document-open-and-extraction
  (with-open [d (pdf/open (sample-pdf))]
    (is (pdf/open? d))
    (is (>= (pdf/page-count d) 1))
    (is (let [t (pdf/extract-text d 0)] (or (.contains t "Hello") (.contains t "Alpha"))))
    (is (seq (pdf/to-markdown d)))
    (is (seq (pdf/to-markdown d 0)))
    (is (.contains (pdf/to-html d) "<"))
    (is (.contains (pdf/to-html d 0) "<"))
    (is (seq (pdf/extract-structured d 0)))
    (is (vector? (pdf/pages d)))))

(deftest open-with-password-and-authenticate
  (with-open [d (pdf/open (sample-pdf))]
    ;; non-encrypted doc: authenticate is callable and returns a boolean.
    (is (boolean? (pdf/authenticate d "anything")))))

(deftest page-element-extraction
  (with-open [d (pdf/open (sample-pdf))]
    (let [pg (pdf/page d 0)
          ws (pdf/words pg)]
      (is (pos? (.width pg)))
      (is (seq ws))
      (is (seq (.text (first ws))))
      (is (vector? (pdf/lines pg)))
      (is (vector? (pdf/chars pg)))
      (is (vector? (pdf/tables pg)))
      (is (vector? (pdf/images pg)))
      (is (vector? (pdf/annotations pg)))
      (is (let [t (pdf/page-text pg)] (or (.contains t "Hello") (.contains t "Alpha"))))
      (is (string? (pdf/page-text pg (BBox. 0.0 0.0 1000.0 1000.0)))))))

(deftest search-and-forms
  (with-open [d (pdf/open (sample-pdf))]
    (let [ms (pdf/search d "Hello")]
      (is (seq ms))
      (is (.contains (.text (first ms)) "Hello")))
    (is (vector? (pdf/form-fields d)))))

(deftest render-page
  (with-open [d (pdf/open (sample-pdf))]
    (is (> (count (pdf/render d 0)) 100))
    (is (> (count (pdf/render d 0 150)) 100))))

(deftest metadata-optional->nil
  (with-open [d (pdf/open (sample-pdf))]
    (is (or (nil? (pdf/producer d)) (string? (pdf/producer d))))
    (is (or (nil? (pdf/creator d)) (string? (pdf/creator d))))))

(deftest document-editor-redaction-and-save
  (with-open [ed (pdf/editor (sample-pdf))]
    (is (pdf/open? ed))
    (pdf/scrub-metadata ed)
    (pdf/add-redaction ed 0 (BBox. 10.0 10.0 50.0 20.0))
    (let [result (pdf/apply-redactions ed)]
      (is (some? result)))
    (is (> (count (pdf/editor-save ed)) 100))))

(deftest auto-extractor
  (with-open [d (pdf/open (sample-pdf))]
    (let [t (pdf/auto-text (pdf/auto-extractor d))]
      (is (or (.contains t "Hello") (.contains t "Alpha"))))))

(deftest explicit-close
  ;; `close` is the escape hatch for non-with-open usage.
  (let [d (pdf/open (sample-pdf))]
    (is (pdf/open? d))
    (pdf/close d)
    (is (not (pdf/open? d)))))
