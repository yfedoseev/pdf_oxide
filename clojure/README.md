# pdf_oxide — Clojure bindings

Idiomatic Clojure bindings — a **thin wrapper** over the mature
[`fyi.oxide:pdf-oxide`](../java) Java binding, which owns the single JNI native
bridge (the `pdf_oxide_jni` crate). This namespace adds **zero native code**: it
calls the Java classes directly via interop and returns Clojure-friendly values
(`java.util.List` → vector, `java.util.Optional` → value-or-`nil`). The handle
types (`Pdf`, `PdfDocument`, `DocumentEditor`) are `AutoCloseable`, so use
`with-open`.

> Why a wrapper, not a separate FFI? Java and Clojure both run on the JVM, so the
> native bridge is written and tested **once** (in the Java binding); Clojure-Java
> interop is trivial.

## Install

Published to [Clojars](https://clojars.org/) as `fyi.oxide/pdf-oxide-clojure`:

```clojure
;; deps.edn
{:deps {fyi.oxide/pdf-oxide-clojure {:mvn/version "0.3.69"}}}
```

```clojure
;; Leiningen
[fyi.oxide/pdf-oxide-clojure "0.3.69"]
```

The JNI native library (`libpdf_oxide_jni`) is not bundled — make it loadable via
`System.loadLibrary("pdf_oxide_jni")` on your `java.library.path`, or point the
Java `NativeLoader` at it with `-Dfyi.oxide.pdf.lib.path=<path>`.

## Build & test

```bash
# 1. build the JNI native library (full feature set)
cargo build --release -p pdf_oxide_jni --features full

# 2. install the Java binding to ~/.m2 (skip the dev profile's Rust rebuild)
( cd java && mvn -P'!dev' -DskipTests install )

# 3. test the Clojure wrapper (JNI lib located via fyi.oxide.pdf.lib.path)
cd clojure
clojure -J-Dfyi.oxide.pdf.lib.path="$PWD/../target/release/libpdf_oxide_jni.so" -M:test
clojure -J-Dfyi.oxide.pdf.lib.path="$PWD/../target/release/libpdf_oxide_jni.so" -M:example
```

The `:test`/`:example` aliases also set `-Djava.library.path=../target/release`,
so `System.loadLibrary("pdf_oxide_jni")` resolves the lib when run from this
directory; pass an absolute `-Dfyi.oxide.pdf.lib.path` to override.

## Use

```clojure
(require '[pdf-oxide.core :as pdf])

(with-open [p (pdf/from-markdown "# Hello\n\nbody\n")
            d (pdf/open (pdf/save p))]
  (println (pdf/page-count d))
  (println (pdf/extract-text d 0))
  (println (pdf/to-markdown d))
  (println (map #(.text %) (pdf/words (pdf/page d 0))))  ; List -> vector
  (println (or (pdf/producer d) "(none)")))              ; Optional -> nil
```

## Layout

```
clojure/
  src/pdf_oxide/core.clj      idiomatic fns over the Java classes (List -> vec, Optional -> nil)
  src/pdf_oxide/example.clj   runnable example (asserted in CI)
  test/pdf_oxide/core_test.clj  coverage over the Java-backed API
  deps.edn
```

## Verification (CI)

`.github/workflows/clojure.yml` on Linux + macOS: build the JNI cdylib → install
the Java binding to local Maven → JDK 17 + Clojure CLI → clj-kondo → `-M:test`
(api-coverage) → run example with an output assertion.
