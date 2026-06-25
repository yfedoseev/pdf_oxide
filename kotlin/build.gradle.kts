// pdf_oxide — Kotlin/JVM (+ Android-ready) bindings.
//
// Thin idiomatic facade over the mature `fyi.oxide:pdf-oxide` Java binding
// (which owns the single JNI native bridge via the `pdf_oxide_jni` crate). This
// module adds ZERO native code: it depends on the Java artifact and layers
// Kotlin sugar (Optional -> nullable, AutoCloseable `use`, fluent helpers).
// The JNI library is loaded by the Java NativeLoader via System.loadLibrary or
// the `-Dfyi.oxide.pdf.lib.path=<libpdf_oxide_jni.so>` override.
plugins {
    kotlin("jvm") version "2.2.20"
    `java-library`
    id("io.gitlab.arturbosch.detekt") version "1.23.8"
    // Publishing to Maven Central via the post-OSSRH Sonatype Central Portal
    // (mirrors the Java binding's central-publishing-maven-plugin setup).
    id("com.vanniktech.maven.publish") version "0.30.0"
}

group = "fyi.oxide"
version = "0.3.69"

repositories {
    mavenCentral()
    mavenLocal() // resolve the locally-installed fyi.oxide:pdf-oxide during dev/CI
}

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
    // The Java binding owns the JNI bridge; we re-export its types (api scope)
    // so Kotlin callers `import fyi.oxide.pdf.*` and get them transitively.
    api("fyi.oxide:pdf-oxide:0.3.69")
    testImplementation(kotlin("test"))
}

kotlin { jvmToolchain(17) }

// Resolve the JNI cdylib (built from the `pdf_oxide_jni` crate) for the Java
// NativeLoader. Override with -DPDF_OXIDE_JNI_LIB=<full path to the .so/.dylib>.
fun jniLibPath(): String =
    System.getProperty("PDF_OXIDE_JNI_LIB")
        ?: System.getenv("PDF_OXIDE_JNI_LIB")
        ?: "${rootDir}/../target/release/libpdf_oxide_jni.so"

tasks.test {
    useJUnitPlatform()
    systemProperty("fyi.oxide.pdf.lib.path", jniLibPath())
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
        description.set("Idiomatic Kotlin/JVM bindings for pdf_oxide — a thin facade over the fyi.oxide:pdf-oxide Java binding (JNI).")
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
    systemProperty("fyi.oxide.pdf.lib.path", jniLibPath())
}

// Shared-scenario regression examples (html / words / tables). Each asserts and
// prints its "<NAME> OK" line; CI greps for that line and fails on non-zero exit.
fun registerExample(taskName: String, main: String) {
    tasks.register<JavaExec>(taskName) {
        group = "application"
        mainClass.set(main)
        classpath = sourceSets.main.get().runtimeClasspath
        systemProperty("fyi.oxide.pdf.lib.path", jniLibPath())
    }
}
registerExample("runHtmlExample", "examples.HtmlExtractionKt")
registerExample("runWordsExample", "examples.WordsGeometryKt")
registerExample("runTablesExample", "examples.TablesExtractionKt")
