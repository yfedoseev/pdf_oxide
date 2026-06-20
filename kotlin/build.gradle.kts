// pdf_oxide — Kotlin/JVM (+ Android-ready) bindings over the C ABI via JNA.
//
// Pure-Kotlin FFI (no native compile): JNA loads libpdf_oxide.{so,dylib,dll} at
// runtime. The native library directory is taken from the `jna.library.path`
// system property or PDF_OXIDE_LIB_DIR (see PdfOxideNative).
plugins {
    kotlin("jvm") version "2.2.20"
    `java-library`
    id("io.gitlab.arturbosch.detekt") version "1.23.8"
}

group = "fyi.oxide"
version = "0.3.68"

repositories { mavenCentral() }

// Static analysis. detekt 1.23.x runs on its own bundled Kotlin analyzer
// (independent of the project's Kotlin 2.2.20), so K2 compatibility is a
// non-issue here. Type-resolution rules are off (no classpath wiring needed);
// the default rule set covers complexity/style/potential-bugs.
detekt {
    source.setFrom("src/main/kotlin", "src/test/kotlin")
    config.setFrom("detekt.yml")
    buildUponDefaultConfig = true
    ignoreFailures = false
}

dependencies {
    implementation("net.java.dev.jna:jna:5.14.0")
    api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
    testImplementation(kotlin("test"))
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
}

kotlin { jvmToolchain(17) }

tasks.test {
    useJUnitPlatform()
    // Point JNA at the freshly built cdylib (override with -DPDF_OXIDE_LIB_DIR=…).
    val libDir = System.getProperty("PDF_OXIDE_LIB_DIR")
        ?: System.getenv("PDF_OXIDE_LIB_DIR")
        ?: "${rootDir}/../target/release"
    systemProperty("jna.library.path", libDir)
    testLogging { events("passed", "failed", "skipped") }
}

// `./gradlew runExample` — runs the smoke example with the cdylib on the path.
tasks.register<JavaExec>("runExample") {
    group = "application"
    mainClass.set("examples.BasicExtractionKt")
    classpath = sourceSets.main.get().runtimeClasspath
    val libDir = System.getProperty("PDF_OXIDE_LIB_DIR")
        ?: System.getenv("PDF_OXIDE_LIB_DIR")
        ?: "${rootDir}/../target/release"
    systemProperty("jna.library.path", libDir)
}
