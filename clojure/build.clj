;; pdf_oxide — Clojure binding build/deploy (clojure.tools.build + deps-deploy).
;;
;;   clojure -T:build jar      ; build target/pdf-oxide-clojure-<version>.jar
;;   clojure -T:build deploy   ; build + deploy to Clojars
;;
;; Coordinates: fyi.oxide/pdf-oxide-clojure. The version is the single canonical
;; string below (kept in lockstep with deps.edn's fyi.oxide/pdf-oxide :mvn/version
;; by the repo-wide scripts/sync_version.py). deps-deploy reads the Clojars creds
;; from the CLOJARS_USERNAME / CLOJARS_PASSWORD environment variables.
(ns build
  (:require [clojure.tools.build.api :as b]
            [deps-deploy.deps-deploy :as dd]))

(def lib 'fyi.oxide/pdf-oxide-clojure)
(def version "0.3.69")
(def class-dir "target/classes")
(def basis (delay (b/create-basis {:project "deps.edn"})))
(def jar-file (format "target/%s-%s.jar" (name lib) version))

(defn clean [_]
  (b/delete {:path "target"}))

(defn jar [_]
  (b/write-pom {:class-dir class-dir
                :lib lib
                :version version
                :basis @basis
                :src-dirs ["src"]
                :pom-data
                [[:description
                  "Idiomatic Clojure bindings for pdf_oxide — a thin wrapper over the fyi.oxide:pdf-oxide Java binding (JNI)."]
                 [:url "https://github.com/yfedoseev/pdf_oxide"]
                 [:licenses
                  [:license
                   [:name "MIT"]
                   [:url "https://opensource.org/licenses/MIT"]]]
                 [:scm
                  [:url "https://github.com/yfedoseev/pdf_oxide"]
                  [:connection "scm:git:https://github.com/yfedoseev/pdf_oxide.git"]
                  [:developerConnection "scm:git:ssh://git@github.com/yfedoseev/pdf_oxide.git"]]]})
  (b/copy-dir {:src-dirs ["src"]
               :target-dir class-dir})
  (b/jar {:class-dir class-dir
          :jar-file jar-file}))

(defn deploy [_]
  (jar nil)
  (dd/deploy {:installer :remote
              :artifact jar-file
              :pom-file (b/pom-path {:lib lib :class-dir class-dir})}))
