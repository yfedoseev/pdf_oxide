cmake_minimum_required(VERSION 3.16)
project(pdf_oxide_cpp VERSION 0.3.69 LANGUAGES CXX)

# Idiomatic C++17 RAII bindings over the pdf_oxide C ABI (header-only wrapper).
#
# Inputs (override on the command line if your layout differs):
#   PDF_OXIDE_INCLUDE_DIR  dir containing  pdf_oxide_c/pdf_oxide.h   (default: ../include)
#   PDF_OXIDE_LIB_DIR      dir containing  libpdf_oxide.{so,dylib}   (default: ../target-wheel/release)
set(PDF_OXIDE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include" CACHE PATH "C header dir")
set(PDF_OXIDE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../target-wheel/release" CACHE PATH "cdylib dir")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Standard install dirs (provides CMAKE_INSTALL_INCLUDEDIR/LIBDIR), used both by
# the INSTALL_INTERFACE include path below and the install() rules at the end.
include(GNUInstallDirs)

# Header-only interface library consumers link against.
add_library(pdf_oxide_cpp INTERFACE)
add_library(pdf_oxide::pdf_oxide_cpp ALIAS pdf_oxide_cpp)
# BUILD_INTERFACE: in-tree paths for examples/tests built here.
# INSTALL_INTERFACE: the installed header layout for find_package() consumers.
target_include_directories(pdf_oxide_cpp INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${PDF_OXIDE_INCLUDE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

find_library(PDF_OXIDE_NATIVE NAMES pdf_oxide libpdf_oxide PATHS "${PDF_OXIDE_LIB_DIR}" NO_DEFAULT_PATH)
if(NOT PDF_OXIDE_NATIVE)
  message(FATAL_ERROR "libpdf_oxide not found in ${PDF_OXIDE_LIB_DIR}; set -DPDF_OXIDE_LIB_DIR=...")
endif()
# Link the native lib only for the in-tree build (examples/tests). The absolute
# path is wrapped in BUILD_INTERFACE so it is NOT baked into the installed/export
# target — find_package() / Conan consumers provide `pdf_oxide` themselves.
target_link_libraries(pdf_oxide_cpp INTERFACE "$<BUILD_INTERFACE:${PDF_OXIDE_NATIVE}>")

option(PDF_OXIDE_BUILD_EXAMPLES "Build C++ examples" ON)
option(PDF_OXIDE_BUILD_TESTS "Build C++ tests" ON)

if(PDF_OXIDE_BUILD_EXAMPLES)
  file(GLOB EXAMPLES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp")
  foreach(ex ${EXAMPLES})
    get_filename_component(name "${ex}" NAME_WE)
    add_executable(example_${name} "${ex}")
    target_link_libraries(example_${name} PRIVATE pdf_oxide_cpp)
  endforeach()
endif()

if(PDF_OXIDE_BUILD_TESTS)
  enable_testing()
  file(GLOB TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
  foreach(t ${TESTS})
    get_filename_component(name "${t}" NAME_WE)
    add_executable(${name} "${t}")
    target_link_libraries(${name} PRIVATE pdf_oxide_cpp)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
      ENVIRONMENT "LD_LIBRARY_PATH=${PDF_OXIDE_LIB_DIR}:$ENV{LD_LIBRARY_PATH}")
  endforeach()
endif()

# ── install / export ────────────────────────────────────────────────────────
# Header-only install so downstream projects can:
#   find_package(pdf_oxide_cpp CONFIG REQUIRED)
#   target_link_libraries(app PRIVATE pdf_oxide::pdf_oxide_cpp)
# The wrapper is headers only; it links the prebuilt native libpdf_oxide at the
# consumer's build time (the consumer points the linker at it, e.g. via
# CMAKE_PREFIX_PATH / link dirs or the Conan package below).
include(CMakePackageConfigHelpers)

# Export only the C++ wrapper's own include dir as a usage requirement; the C
# header dir and native lib path are environment-specific and supplied by the
# consumer, so they are not baked into the exported target.
install(TARGETS pdf_oxide_cpp
  EXPORT pdf_oxide_cppTargets
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.hpp")

set(PDF_OXIDE_CPP_INSTALL_CMAKEDIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/pdf_oxide_cpp"
  CACHE STRING "Install location for pdf_oxide_cpp CMake package files")

install(EXPORT pdf_oxide_cppTargets
  FILE pdf_oxide_cppTargets.cmake
  NAMESPACE pdf_oxide::
  DESTINATION "${PDF_OXIDE_CPP_INSTALL_CMAKEDIR}")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pdf_oxide_cppConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/pdf_oxide_cppConfig.cmake"
  INSTALL_DESTINATION "${PDF_OXIDE_CPP_INSTALL_CMAKEDIR}")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/pdf_oxide_cppConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/pdf_oxide_cppConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/pdf_oxide_cppConfigVersion.cmake"
  DESTINATION "${PDF_OXIDE_CPP_INSTALL_CMAKEDIR}")
