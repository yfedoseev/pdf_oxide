;; words_geometry — build a PDF from Markdown, then extract word geometry.
;; Shared-scenario regression example run in CI (no external fixture). An
;; uncaught assertion exits non-zero; prints "WORDS OK" on success. Uses the
;; Java-backed API through the Clojure facade (`with-open` on the handles).
(ns pdf-oxide.example-words
  (:require [pdf-oxide.core :as pdf]))

(defn -main [& _]
  (with-open [p (pdf/from-markdown "# Hello pdf_oxide\n\nThis is a **Clojure** regression example.\n")
              d (pdf/open (pdf/save p))]
    (let [words (pdf/words (pdf/page d 0))
          first-word (first words)]
      (assert (seq words) "words_geometry: expected at least one word")
      (println "words:" (count words) "; first:" (.text first-word) "@" (.bbox first-word))
      (assert (= "Hello" (.text first-word)) "words_geometry: expected first word 'Hello'")
      (assert (>= (.width (.bbox first-word)) 0.0) "words_geometry: expected first word to have a bbox")
      (println "WORDS OK"))))
