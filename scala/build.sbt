// pdf_oxide — Scala bindings over the C ABI via JNA (same mechanism as Kotlin).
ThisBuild / organization := "fyi.oxide"
ThisBuild / version := "0.3.68"
ThisBuild / scalaVersion := "3.3.4"

lazy val root = (project in file("."))
  .settings(
    name := "pdf-oxide-scala",
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
