;; basic_extraction — build a PDF from Markdown, then extract it back.
;; Run in CI as a smoke example (no external fixture).
(ns pdf-oxide.example
  (:require [pdf-oxide.core :as pdf]))

(defn -main [& _]
  (with-open [p (pdf/from-markdown "# Hello pdf_oxide\n\nThis is a **Clojure** binding smoke example.\n")
              d (pdf/open-from-bytes (pdf/to-bytes p))]
    (println "pages:  " (pdf/page-count d))
    (println "version:" (let [v (pdf/version d)] (str (:major v) "." (:minor v))))
    (println "--- text (page 0) ---")
    (println (pdf/extract-text d 0))
    (println "--- markdown (all) ---")
    (println (pdf/to-markdown-all d))))
