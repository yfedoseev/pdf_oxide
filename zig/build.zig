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

    // Shared configuration applied to every artifact.
    const configure = struct {
        fn apply(step: *std.Build.Step.Compile, inc: []const u8, lib: []const u8) void {
            step.addIncludePath(.{ .cwd_relative = inc });
            step.addLibraryPath(.{ .cwd_relative = lib });
            step.linkSystemLibrary("pdf_oxide");
            step.linkLibC();
        }
    }.apply;

    // The importable module (lib/pdf_oxide.zig).
    const mod = b.addModule("pdf_oxide", .{
        .root_source_file = b.path("lib/pdf_oxide.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addIncludePath(.{ .cwd_relative = include_dir });

    // ── tests (api-coverage) ──────────────────────────────────────────────
    const tests = b.addTest(.{
        .root_source_file = b.path("lib/pdf_oxide.zig"),
        .target = target,
        .optimize = optimize,
    });
    configure(tests, include_dir, lib_dir);
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run api-coverage tests").dependOn(&run_tests.step);

    // ── example (smoke) ───────────────────────────────────────────────────
    const example = b.addExecutable(.{
        .name = "basic_extraction",
        .root_source_file = b.path("examples/basic_extraction.zig"),
        .target = target,
        .optimize = optimize,
    });
    example.root_module.addImport("pdf_oxide", mod);
    configure(example, include_dir, lib_dir);
    const run_example = b.addRunArtifact(example);
    b.step("example", "Run the smoke example").dependOn(&run_example.step);
}
