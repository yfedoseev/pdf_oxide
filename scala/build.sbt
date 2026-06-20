// pdf_oxide — Scala bindings over the C ABI via JNA (same mechanism as Kotlin).
ThisBuild / organization := "fyi.oxide"
ThisBuild / organizationName := "PDF Oxide"
ThisBuild / version := "0.3.68"
ThisBuild / scalaVersion := "3.3.4"

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
    description := "Idiomatic Scala 3 bindings for pdf_oxide — fast PDF text/Markdown/HTML extraction over the C ABI via JNA.",
    libraryDependencies ++= Seq(
      "net.java.dev.jna" % "jna" % "5.14.0",
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    ),
    // JNA finds the freshly built cdylib via jna.library.path.
    Test / javaOptions += {
      val dir = sys.props.getOrElse(
        "PDF_OXIDE_LIB_DIR",
        sys.env.getOrElse("PDF_OXIDE_LIB_DIR", s"${baseDirectory.value}/../target/release")
      )
      s"-Djna.library.path=$dir"
    },
    Test / fork := true
  )
