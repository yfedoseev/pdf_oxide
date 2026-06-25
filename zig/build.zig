// pdf_oxide — Zig bindings build.
//
// First-class C interop: the module @cImports include/pdf_oxide_c/pdf_oxide.h
// and links the default-feature cdylib (libpdf_oxide). Paths are taken from
// -DPDF_OXIDE_INCLUDE_DIR / -DPDF_OXIDE_LIB_DIR (defaults: ../include,
// ../target/release).
//
// Targets: `zig build test` (api-coverage), `zig build example` (smoke example).
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const include_dir = b.option([]const u8, "PDF_OXIDE_INCLUDE_DIR", "C header dir") orelse "../include";
    const lib_dir = b.option([]const u8, "PDF_OXIDE_LIB_DIR", "cdylib dir") orelse "../target/release";

    // Apply the C header + cdylib link settings to a module (Zig 0.15 moved
    // these from the Compile step onto Module).
    const wire = struct {
        fn apply(m: *std.Build.Module, inc: []const u8, lib: []const u8) void {
            m.addIncludePath(.{ .cwd_relative = inc });
            m.addLibraryPath(.{ .cwd_relative = lib });
            m.linkSystemLibrary("pdf_oxide", .{});
            m.link_libc = true;
        }
    }.apply;

    // The importable module (lib/pdf_oxide.zig).
    const mod = b.addModule("pdf_oxide", .{
        .root_source_file = b.path("lib/pdf_oxide.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addIncludePath(.{ .cwd_relative = include_dir });

    // ── tests (api-coverage) — root module is the lib itself ──────────────
    const test_mod = b.createModule(.{
        .root_source_file = b.path("lib/pdf_oxide.zig"),
        .target = target,
        .optimize = optimize,
    });
    wire(test_mod, include_dir, lib_dir);
    const tests = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run api-coverage tests").dependOn(&run_tests.step);

    // ── example (smoke) ───────────────────────────────────────────────────
    const ex_mod = b.createModule(.{
        .root_source_file = b.path("examples/basic_extraction.zig"),
        .target = target,
        .optimize = optimize,
    });
    ex_mod.addImport("pdf_oxide", mod);
    wire(ex_mod, include_dir, lib_dir);
    const example = b.addExecutable(.{ .name = "basic_extraction", .root_module = ex_mod });
    const run_example = b.addRunArtifact(example);
    b.step("example", "Run the smoke example").dependOn(&run_example.step);

    // ── shared-scenario examples (mirrored across language bindings) ───────
    const Scenario = struct { step: []const u8, src: []const u8, name: []const u8, desc: []const u8 };
    const scenarios = [_]Scenario{
        .{ .step = "example-html", .src = "examples/html_extraction.zig", .name = "html_extraction", .desc = "Run the html_extraction example" },
        .{ .step = "example-words", .src = "examples/words_geometry.zig", .name = "words_geometry", .desc = "Run the words_geometry example" },
        .{ .step = "example-tables", .src = "examples/tables_extraction.zig", .name = "tables_extraction", .desc = "Run the tables_extraction example" },
    };
    for (scenarios) |sc| {
        const sc_mod = b.createModule(.{
            .root_source_file = b.path(sc.src),
            .target = target,
            .optimize = optimize,
        });
        sc_mod.addImport("pdf_oxide", mod);
        wire(sc_mod, include_dir, lib_dir);
        const sc_exe = b.addExecutable(.{ .name = sc.name, .root_module = sc_mod });
        const sc_run = b.addRunArtifact(sc_exe);
        b.step(sc.step, sc.desc).dependOn(&sc_run.step);
    }
}
