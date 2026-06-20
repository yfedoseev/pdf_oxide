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

let package = Package(
    name: "PdfOxide",
    products: [
        .library(name: "PdfOxide", targets: ["PdfOxide"]),
        .executable(name: "basic_extraction", targets: ["Example"]),
    ],
    targets: [
        // System-library target: wraps the C header (module.modulemap).
        .systemLibrary(name: "CPdfOxide", path: "Sources/CPdfOxide"),
        .target(
            name: "PdfOxide",
            dependencies: ["CPdfOxide"],
            cSettings: [.unsafeFlags(["-I", includeDir])],
            swiftSettings: [.unsafeFlags(["-I", includeDir])],
            linkerSettings: [
                .unsafeFlags(["-L", libDir, "-lpdf_oxide", "-Xlinker", "-rpath", "-Xlinker", libDir])
            ]
        ),
        .executableTarget(
            name: "Example",
            dependencies: ["PdfOxide"],
            path: "Sources/Example"
        ),
        .testTarget(
            name: "PdfOxideTests",
            dependencies: ["PdfOxide"]
        ),
    ]
)
