// pdf_oxide — Scala 3 bindings: a thin idiomatic facade over the
// `fyi.oxide:pdf-oxide` Java binding (which owns the single JNI native bridge).
// No native code here — depend on the Java artifact and add Scala sugar
// (Optional -> Option, java.util.List -> Seq, Using on AutoCloseable handles).
ThisBuild / organization := "fyi.oxide"
ThisBuild / organizationName := "PDF Oxide"
ThisBuild / version := "0.3.69"
ThisBuild / scalaVersion := "3.3.4"

// scalafix needs SemanticDB. On Scala 3 it's emitted by the compiler itself
// (sbt-scalafix wires `-Xsemanticdb` via semanticdbEnabled); no extra compiler
// plugin/version pin is required as it is on Scala 2.
ThisBuild / semanticdbEnabled := true

// Maven Central publishing via the post-OSSRH Sonatype Central Portal
// (mirrors the Java binding). sbt-ci-release + sbt-sonatype target
// central.sonatype.com; credentials + PGP key come from CI env
// (SONATYPE_USERNAME/PASSWORD, PGP_SECRET/PGP_PASSPHRASE).
ThisBuild / homepage := Some(url("https://github.com/yfedoseev/pdf_oxide"))
ThisBuild / licenses := Seq("MIT" -> url("https://opensource.org/licenses/MIT"))
ThisBuild / developers := List(
  Developer("yfedoseev", "Yury Fedoseev", "yfedoseev@gmail.com", url("https://github.com/yfedoseev"))
)
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/yfedoseev/pdf_oxide"),
    "scm:git:https://github.com/yfedoseev/pdf_oxide.git"
  )
)
// Route to the Central Portal host (post-OSSRH).
ThisBuild / sonatypeCredentialHost := "central.sonatype.com"
ThisBuild / sonatypeProfileName := "fyi.oxide"

lazy val root = (project in file("."))
  .settings(
    name := "pdf-oxide-scala",
    description := "Idiomatic Scala 3 bindings for pdf_oxide — a thin facade over the fyi.oxide:pdf-oxide Java binding (JNI).",
    resolvers += Resolver.mavenLocal, // resolve the locally-installed Java artifact during dev/CI
    libraryDependencies ++= Seq(
      "fyi.oxide" % "pdf-oxide" % "0.3.69",
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    ),
    // The Java NativeLoader resolves the JNI cdylib via this property; override
    // with -DPDF_OXIDE_JNI_LIB=<full path to libpdf_oxide_jni.so/.dylib>.
    Test / javaOptions += {
      val so = sys.props.getOrElse(
        "PDF_OXIDE_JNI_LIB",
        sys.env.getOrElse("PDF_OXIDE_JNI_LIB", s"${baseDirectory.value}/../target/release/libpdf_oxide_jni.so")
      )
      s"-Dfyi.oxide.pdf.lib.path=$so"
    },
    Test / fork := true
  )
