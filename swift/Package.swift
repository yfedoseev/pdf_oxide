// swift-tools-version:5.9
// pdf_oxide — Swift bindings over the C ABI.
//
// CPdfOxide is a system-library target exposing the cbindgen C header via a
// module map; PdfOxide is the idiomatic Swift wrapper. The native cdylib
// (libpdf_oxide) and the header dir are located via -L/-I unsafe flags pointing
// at PDF_OXIDE_LIB_DIR / PDF_OXIDE_INCLUDE_DIR (defaults ../target/release,
// ../include) — override in CI.
import PackageDescription
import Foundation

let env = ProcessInfo.processInfo.environment
let libDir = env["PDF_OXIDE_LIB_DIR"] ?? "../target/release"
let includeDir = env["PDF_OXIDE_INCLUDE_DIR"] ?? "../include"

// The CPdfOxide system-library module's shim.h does
// `#include <pdf_oxide_c/pdf_oxide.h>`, which the Clang module importer resolves
// — so the header dir must reach Clang via `-Xcc -I`, not swiftc's own `-I`
// (the latter only adds a Swift-module search path). Every target that imports
// PdfOxide (which re-exports CPdfOxide) rebuilds that Clang module, so the flag
// has to be on each of them.
let headerSearch: [SwiftSetting] = [.unsafeFlags(["-Xcc", "-I\(includeDir)"])]

let package = Package(
    name: "PdfOxide",
    products: [
        .library(name: "PdfOxide", targets: ["PdfOxide"]),
        .executable(name: "basic_extraction", targets: ["Example"]),
        .executable(name: "html_extraction", targets: ["HtmlExtraction"]),
        .executable(name: "words_geometry", targets: ["WordsGeometry"]),
        .executable(name: "tables_extraction", targets: ["TablesExtraction"]),
    ],
    targets: [
        // System-library target: wraps the C header (module.modulemap).
        .systemLibrary(name: "CPdfOxide", path: "Sources/CPdfOxide"),
        .target(
            name: "PdfOxide",
            dependencies: ["CPdfOxide"],
            cSettings: [.unsafeFlags(["-I", includeDir])],
            swiftSettings: headerSearch,
            linkerSettings: [
                .unsafeFlags(["-L", libDir, "-lpdf_oxide", "-Xlinker", "-rpath", "-Xlinker", libDir])
            ]
        ),
        .executableTarget(
            name: "Example",
            dependencies: ["PdfOxide"],
            path: "Sources/Example",
            swiftSettings: headerSearch
        ),
        .executableTarget(
            name: "HtmlExtraction",
            dependencies: ["PdfOxide"],
            path: "Sources/HtmlExtraction",
            swiftSettings: headerSearch
        ),
        .executableTarget(
            name: "WordsGeometry",
            dependencies: ["PdfOxide"],
            path: "Sources/WordsGeometry",
            swiftSettings: headerSearch
        ),
        .executableTarget(
            name: "TablesExtraction",
            dependencies: ["PdfOxide"],
            path: "Sources/TablesExtraction",
            swiftSettings: headerSearch
        ),
        .testTarget(
            name: "PdfOxideTests",
            dependencies: ["PdfOxide"],
            swiftSettings: headerSearch
        ),
    ]
)
