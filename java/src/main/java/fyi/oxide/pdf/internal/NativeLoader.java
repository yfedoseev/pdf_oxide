/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.internal;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads the {@code pdf_oxide_jni} native library exactly once per JVM.
 *
 * <p>This class is package-public to {@code fyi.oxide.pdf.*} but is
 * considered internal API: invoke it indirectly by referencing any
 * public class in {@code fyi.oxide.pdf} (e.g. {@code PdfDocument})
 * — each one carries a {@code static { NativeLoader.ensureLoaded(); }}
 * initialiser, and the CAS guard makes the order of class-loading
 * irrelevant.
 *
 * <p><b>Resolution order</b> (first match wins):
 * <ol>
 *   <li>{@code -Dfyi.oxide.pdf.lib.path=<absolute path>} — explicit
 *       override; loaded via {@link System#load(String)}.</li>
 *   <li>{@code -Dfyi.oxide.pdf.use.systemlib=true} — loaded via
 *       {@link System#loadLibrary(String)} ({@code pdf_oxide_jni}).</li>
 *   <li>Bundled resource at
 *       {@code /fyi/oxide/pdf/native/<OS>/<ARCH>/<libname>} — extracted
 *       to a UUID-suffixed temp file (multi-classloader safe — without
 *       the UUID, two web apps in the same JVM hit
 *       {@code UnsatisfiedLinkError}; see Apache Flink FLINK-5408 for
 *       the prior art) and loaded via {@link System#load(String)}.</li>
 * </ol>
 *
 * <p><b>Supported {@code <OS>/<ARCH>} pairs</b> in v0.3.53:
 * {@code Linux/x86_64}, {@code Linux/aarch64}, {@code Linux/x86_64-musl}
 * (Alpine; feature-gated build), {@code Mac/x86_64}, {@code Mac/aarch64},
 * {@code Windows/x86_64}.
 *
 * <p><b>Tunables</b>:
 * <ul>
 *   <li>{@code -Dfyi.oxide.pdf.tempdir=<dir>} — overrides
 *       {@code java.io.tmpdir} for the extraction step. Useful in
 *       Docker non-root, Kubernetes read-only-root-filesystem, and
 *       FIPS-locked-tmp environments.</li>
 * </ul>
 *
 * <p><b>macOS note</b>: extracted {@code .dylib} files may be tagged
 * with the {@code com.apple.quarantine} xattr if the JAR was downloaded
 * by a browser. {@link System#load(String)} then fails with a cryptic
 * dlopen error. Either use the {@code -Dfyi.oxide.pdf.lib.path}
 * override or strip the xattr with {@code xattr -d com.apple.quarantine}.
 * Maven/Gradle dependency-resolution downloads don't tag the JAR.
 *
 * <p>See the v0.3.53 release plan
 * {@code docs/releases/plans/v0.3.53/00-common-foundation.md} §3
 * for the full native-loader contract.
 */
public final class NativeLoader {

    private static final Logger LOG = LoggerFactory.getLogger(NativeLoader.class);

    /** Library base name; {@link System#mapLibraryName(String)} resolves it. */
    static final String LIB_NAME = "pdf_oxide_jni";

    /** Java package-rooted resource prefix for bundled natives. */
    static final String NATIVE_RESOURCE_ROOT = "/fyi/oxide/pdf/native";

    /** Implementation version; bumped lockstep with Cargo / Maven. */
    static final String VERSION = "0.3.53";

    /** System property: full path to a native library to load directly. */
    static final String PROP_LIB_PATH = "fyi.oxide.pdf.lib.path";

    /** System property: opt into {@link System#loadLibrary(String)}. */
    static final String PROP_USE_SYSTEM_LIB = "fyi.oxide.pdf.use.systemlib";

    /** System property: override the temp directory for resource extraction. */
    static final String PROP_TEMP_DIR = "fyi.oxide.pdf.tempdir";

    /** Single-shot guard. CAS prevents re-loading on concurrent class init. */
    private static final AtomicBoolean LOADED = new AtomicBoolean(false);

    private NativeLoader() {
        // Static-only.
    }

    /**
     * Loads the native library on first invocation; subsequent calls
     * are no-ops. Idempotent and thread-safe.
     *
     * @throws UnsatisfiedLinkError if the native library cannot be
     *         located or loaded. Wraps the underlying cause (IOException,
     *         dlopen failure, etc.) in the error's cause chain.
     */
    public static void ensureLoaded() {
        if (!LOADED.compareAndSet(false, true)) {
            return;
        }
        try {
            doLoad();
        } catch (RuntimeException | Error e) {
            // Reset the guard so a retry is possible (e.g. user fixes
            // the temp-dir permissions and re-invokes). Production
            // callers will usually never retry, but tests want this.
            LOADED.set(false);
            throw e;
        }
    }

    private static void doLoad() {
        // 1. Explicit override.
        final String overridePath = System.getProperty(PROP_LIB_PATH);
        if (overridePath != null && !overridePath.isEmpty()) {
            LOG.debug("Loading pdf_oxide_jni from -D{}={}", PROP_LIB_PATH, overridePath);
            System.load(overridePath);
            return;
        }

        // 2. System library opt-in.
        if (Boolean.getBoolean(PROP_USE_SYSTEM_LIB)) {
            LOG.debug("Loading pdf_oxide_jni via System.loadLibrary({})", LIB_NAME);
            System.loadLibrary(LIB_NAME);
            return;
        }

        // 3. Bundled resource — extract + load.
        loadBundled();
    }

    private static void loadBundled() {
        final String osDir = detectOsDir();
        final String archDir = detectArchDir();
        final String libFileName = System.mapLibraryName(LIB_NAME);
        final String resourcePath = String.join("/", NATIVE_RESOURCE_ROOT, osDir, archDir, libFileName);

        LOG.debug("Loading pdf_oxide_jni from JAR resource: {}", resourcePath);

        final Path tempDir = resolveTempDir();
        final Path tmp = tempDir.resolve("pdf-oxide-" + VERSION + "-" + UUID.randomUUID() + "-" + libFileName);

        try (InputStream in = NativeLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new UnsatisfiedLinkError("No bundled pdf_oxide_jni for " + osDir + "/" + archDir
                        + " (resource " + resourcePath + " not in JAR). "
                        + "Use -D" + PROP_LIB_PATH + "=<path> to point at a "
                        + "locally-built library, or -D" + PROP_USE_SYSTEM_LIB
                        + "=true to load from the system path.");
            }
            Files.createDirectories(tempDir);
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            tmp.toFile().setExecutable(true);
            tmp.toFile().deleteOnExit();
        } catch (IOException e) {
            UnsatisfiedLinkError err =
                    new UnsatisfiedLinkError("Failed to extract pdf_oxide_jni to " + tmp + ": " + e.getMessage());
            err.initCause(e);
            throw err;
        }

        try {
            System.load(tmp.toAbsolutePath().toString());
        } catch (UnsatisfiedLinkError e) {
            // Annotate with the macOS-quarantine hint when applicable.
            if (osDir.equals("Mac") && e.getMessage() != null && e.getMessage().contains("dlopen")) {
                UnsatisfiedLinkError annotated =
                        new UnsatisfiedLinkError(e.getMessage() + " — if you downloaded the JAR via a browser, "
                                + "remove the quarantine xattr: "
                                + "xattr -d com.apple.quarantine " + tmp
                                + ", or use -D" + PROP_LIB_PATH + "=<path>.");
                annotated.initCause(e);
                throw annotated;
            }
            throw e;
        }
    }

    /** Resolve the temp directory honoring the override knob. */
    private static Path resolveTempDir() {
        final String override = System.getProperty(PROP_TEMP_DIR);
        if (override != null && !override.isEmpty()) {
            return Paths.get(override);
        }
        return Paths.get(System.getProperty("java.io.tmpdir"));
    }

    /**
     * Map {@code os.name} into the bundled-resource OS segment.
     * Returns one of {@code Linux}, {@code Mac}, {@code Windows}.
     */
    static String detectOsDir() {
        final String osName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (osName.startsWith("linux")) {
            return "Linux";
        }
        if (osName.startsWith("mac") || osName.contains("darwin")) {
            return "Mac";
        }
        if (osName.startsWith("windows")) {
            return "Windows";
        }
        throw new UnsatisfiedLinkError(
                "Unsupported OS: " + System.getProperty("os.name") + ". v0.3.53 ships natives for Linux/Mac/Windows.");
    }

    /**
     * Map {@code os.arch} into the bundled-resource ARCH segment.
     * Returns one of {@code x86_64}, {@code aarch64}, optionally with
     * a {@code -musl} suffix on Alpine Linux (detected via the
     * {@code java.vm.vendor} hint when available).
     */
    static String detectArchDir() {
        final String osArch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        final String arch;
        if (osArch.equals("amd64") || osArch.equals("x86_64") || osArch.equals("x64")) {
            arch = "x86_64";
        } else if (osArch.equals("aarch64") || osArch.equals("arm64")) {
            arch = "aarch64";
        } else {
            throw new UnsatisfiedLinkError("Unsupported architecture: " + System.getProperty("os.arch")
                    + ". v0.3.53 ships x86_64 and aarch64 natives.");
        }

        // musl detection on Linux: best-effort. Users on Alpine /
        // distroless-musl images can also opt in explicitly via
        // -Dfyi.oxide.pdf.tempdir + -Dfyi.oxide.pdf.lib.path. The
        // /etc/os-release check below is intentionally cheap and may
        // false-negative on minimal containers; that's acceptable
        // because the override knob covers them.
        if ("x86_64".equals(arch) && "Linux".equals(detectOsDir()) && isMusl()) {
            return "x86_64-musl";
        }
        return arch;
    }

    /**
     * Best-effort musl detection. Reads {@code /etc/os-release} and
     * looks for {@code alpine} as the ID. Returns false on any error
     * (treating glibc as the safe default — the override knob is the
     * escape hatch).
     */
    private static boolean isMusl() {
        try {
            final Path osRelease = Paths.get("/etc/os-release");
            if (!Files.isReadable(osRelease)) {
                return false;
            }
            for (String line : Files.readAllLines(osRelease)) {
                final String lower = line.toLowerCase(Locale.ROOT);
                if (lower.startsWith("id=alpine") || lower.startsWith("id=\"alpine\"")) {
                    return true;
                }
            }
            return false;
        } catch (IOException e) {
            return false;
        }
    }
}
