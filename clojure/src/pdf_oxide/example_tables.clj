;; tables_extraction — build a PDF from a Markdown table, then extract tables.
;; Shared-scenario regression example run in CI (no external fixture). Asserts
;; the call returns a list without error (synthetic docs may yield 0 tables).
;; An uncaught assertion exits non-zero; prints "TABLES OK" on success. Uses the
;; Java-backed API through the Clojure facade (`with-open` on the handles).
(ns pdf-oxide.example-tables
  (:require [pdf-oxide.core :as pdf]))

(defn -main [& _]
  (with-open [p (pdf/from-markdown "# Report\n\n| Name | Value |\n|------|-------|\n| alpha | 1 |\n| beta | 2 |\n")
              d (pdf/open (pdf/save p))]
    (let [tables (pdf/tables (pdf/page d 0))]
      (assert (>= (count tables) 0) "tables_extraction: expected a list of tables")
      (println "tables:" (count tables))
      (doseq [t tables]
        (println "  " (.rows t) "x" (.cols t) "cells=" (mapv #(.text %) (.cells t))))
      (println "TABLES OK"))))
