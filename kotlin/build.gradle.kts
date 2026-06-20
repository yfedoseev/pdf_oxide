// pdf_oxide — Kotlin/JVM (+ Android-ready) bindings over the C ABI via JNA.
//
// Pure-Kotlin FFI (no native compile): JNA loads libpdf_oxide.{so,dylib,dll} at
// runtime. The native library directory is taken from the `jna.library.path`
// system property or PDF_OXIDE_LIB_DIR (see PdfOxideNative).
plugins {
    kotlin("jvm") version "2.2.20"
    `java-library`
    id("io.gitlab.arturbosch.detekt") version "1.23.8"
    // Publishing to Maven Central via the post-OSSRH Sonatype Central Portal
    // (mirrors the Java binding's central-publishing-maven-plugin setup).
    id("com.vanniktech.maven.publish") version "0.30.0"
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

// Maven Central publishing (Sonatype Central Portal). Credentials + signing key
// come from CI env (ORG_GRADLE_PROJECT_mavenCentralUsername / *Password /
// signingInMemoryKey / *Password), same secrets family as the Java binding.
// GPG-signs all publications; autoPublish is left to the release-gate workflow.
mavenPublishing {
    publishToMavenCentral(com.vanniktech.maven.publish.SonatypeHost.CENTRAL_PORTAL, automaticRelease = false)
    signAllPublications()
    coordinates("fyi.oxide", "pdf-oxide-kotlin", version.toString())
    pom {
        name.set("pdf_oxide Kotlin bindings")
        description.set("Idiomatic Kotlin/JVM bindings for pdf_oxide — fast PDF text/Markdown/HTML extraction over the C ABI via JNA.")
        url.set("https://github.com/yfedoseev/pdf_oxide")
        licenses {
            license {
                name.set("MIT")
                url.set("https://opensource.org/licenses/MIT")
            }
        }
        developers {
            developer {
                id.set("yfedoseev")
                name.set("Yury Fedoseev")
                email.set("yfedoseev@gmail.com")
            }
        }
        scm {
            url.set("https://github.com/yfedoseev/pdf_oxide")
            connection.set("scm:git:https://github.com/yfedoseev/pdf_oxide.git")
            developerConnection.set("scm:git:ssh://git@github.com/yfedoseev/pdf_oxide.git")
        }
    }
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
