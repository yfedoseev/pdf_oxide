;; Coverage for the Clojure facade over the Java binding. Self-contained: builds
;; its own PDF from Markdown, then exercises the main entry points. Java handles
;; are AutoCloseable -> `with-open`.
(ns pdf-oxide.core-test
  (:require [clojure.test :refer [deftest is]]
            [pdf-oxide.core :as pdf]))

(defn sample-pdf ^bytes []
  (with-open [p (pdf/from-markdown "# Alpha Heading\n\nHello world from the Clojure facade. Beta gamma.\n")]
    (pdf/save p)))

(deftest pdf-from-markdown-and-save
  (let [bytes (sample-pdf)]
    (is (> (count bytes) 100))
    (is (= (byte \%) (aget bytes 0)))))

(deftest document-open-and-extraction
  (with-open [d (pdf/open (sample-pdf))]
    (is (pdf/open? d))
    (is (>= (pdf/page-count d) 1))
    (let [t (pdf/extract-text d 0)]
      (is (or (.contains t "Hello") (.contains t "Alpha"))))
    (is (seq (pdf/to-markdown d)))
    (is (.contains (pdf/to-html d) "<"))))

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
      (is (vector? (pdf/annotations pg))))))

(deftest search-and-forms
  (with-open [d (pdf/open (sample-pdf))]
    (let [ms (pdf/search d "Hello")]
      (is (seq ms))
      (is (.contains (.text (first ms)) "Hello")))
    (is (vector? (pdf/form-fields d)))))

(deftest render-page
  (with-open [d (pdf/open (sample-pdf))]
    (is (> (count (pdf/render d 0)) 100))))

(deftest metadata-optional->nil
  (with-open [d (pdf/open (sample-pdf))]
    ;; Optional.empty -> nil; just assert the accessors are callable.
    (is (or (nil? (pdf/producer d)) (string? (pdf/producer d))))
    (is (or (nil? (pdf/creator d)) (string? (pdf/creator d))))))

(deftest document-editor-round-trip
  (with-open [ed (pdf/editor (sample-pdf))]
    (is (pdf/open? ed))
    (pdf/scrub-metadata ed)
    (is (> (count (pdf/editor-save ed)) 100))))

(deftest auto-extractor
  (with-open [d (pdf/open (sample-pdf))]
    (let [t (pdf/auto-text (pdf/auto-extractor d))]
      (is (or (.contains t "Hello") (.contains t "Alpha"))))))
