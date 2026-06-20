;; basic_extraction — build a PDF from Markdown, then extract it back.
;; Run in CI as a smoke example (no external fixture). Uses the Java-backed API
;; through the Clojure facade (`with-open` on the AutoCloseable handles).
(ns pdf-oxide.example
  (:require [pdf-oxide.core :as pdf]))

(defn -main [& _]
  (with-open [p (pdf/from-markdown "# Hello pdf_oxide\n\nThis is a **Clojure** binding smoke example.\n")
              d (pdf/open (pdf/save p))]
    (println "pages:   " (pdf/page-count d))
    (println "producer:" (or (pdf/producer d) "(none)"))
    (println "--- text (page 0) ---")
    (println (pdf/extract-text d 0))
    (println "--- markdown (all) ---")
    (println (pdf/to-markdown d))
    (println "--- words (page 0) ---")
    (doseq [w (take 8 (pdf/words (pdf/page d 0)))]
      (println "  " (.text w) "@" (.bbox w)))))
