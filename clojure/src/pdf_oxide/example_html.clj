;; html_extraction — build a PDF from Markdown, then extract it back to HTML.
;; Shared-scenario regression example run in CI (no external fixture). An
;; uncaught assertion exits non-zero; prints "HTML OK" on success. Uses the
;; Java-backed API through the Clojure facade (`with-open` on the handles).
(ns pdf-oxide.example-html
  (:require [pdf-oxide.core :as pdf]))

(defn -main [& _]
  (with-open [p (pdf/from-markdown "# Hello pdf_oxide\n\nThis is a **Clojure** regression example.\n")
              d (pdf/open (pdf/save p))]
    (let [html (pdf/to-html d)]
      (println html)
      (assert (.contains html "<") "html_extraction: expected HTML to contain '<'")
      (assert (.contains html "pdf_oxide") "html_extraction: expected HTML to contain 'pdf_oxide'")
      (println "HTML OK"))))
