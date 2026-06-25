# Conan 2.x recipe for the header-only pdf_oxide C++ wrapper.
#
# This packages ONLY the C++ RAII headers (include/pdf_oxide/*.hpp) plus the C
# ABI header they include (include/pdf_oxide_c/pdf_oxide.h, taken from the repo
# root ../include). It is header-only: there is nothing to compile here.
#
# IMPORTANT: the wrapper links the prebuilt native `pdf_oxide` shared library at
# the consumer's build/link time. That native lib is produced by the Rust crate
# (`cargo build --release --lib --features ...`) and is NOT built or shipped by
# this recipe — the consumer must make `libpdf_oxide.{so,dylib,dll}` available
# on the linker search path (e.g. install it into the Conan package's lib dir,
# or pass its location via your toolchain). package_info() declares the
# `pdf_oxide` system lib so consumers automatically get `-lpdf_oxide`.
#
# A vcpkg port for this wrapper lives upstream in microsoft/vcpkg and is
# submitted/updated manually (out of scope for this recipe).

import os

from conan import ConanFile
from conan.tools.files import copy


class PdfOxideCppConan(ConanFile):
    name = "pdf_oxide_cpp"
    version = "0.3.69"
    license = "MIT"
    url = "https://github.com/yfedoseev/pdf_oxide"
    homepage = "https://github.com/yfedoseev/pdf_oxide"
    description = "Idiomatic, header-only C++17 RAII bindings over the pdf_oxide C ABI."
    topics = ("pdf", "text-extraction", "markdown", "header-only", "ffi")

    # Header-only: no settings affect the package contents, no build step.
    package_type = "header-library"
    no_copy_source = True

    # Ship both the C++ wrapper headers (cpp/include) and the C ABI header that
    # they #include (repo-root include). recipe_folder is cpp/.
    exports_sources = "include/*", "../include/*", "README.md"

    def package_id(self):
        self.info.clear()  # header-only: identical package for all configs

    def package(self):
        # C++ RAII wrapper headers.
        copy(
            self,
            "*.hpp",
            src=os.path.join(self.source_folder, "include"),
            dst=os.path.join(self.package_folder, "include"),
        )
        # C ABI header (pdf_oxide_c/pdf_oxide.h) from the repo root.
        copy(
            self,
            "*.h",
            src=os.path.join(self.source_folder, os.pardir, "include"),
            dst=os.path.join(self.package_folder, "include"),
        )

    def package_info(self):
        # Header-only: no own libs to link, but expose the include dir and the
        # native pdf_oxide shared lib so consumers link `-lpdf_oxide`.
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.system_libs = ["pdf_oxide"]
        # Match the CMake package: find_package(pdf_oxide_cpp) +
        # target pdf_oxide::pdf_oxide_cpp.
        self.cpp_info.set_property("cmake_file_name", "pdf_oxide_cpp")
        self.cpp_info.set_property("cmake_target_name", "pdf_oxide::pdf_oxide_cpp")
