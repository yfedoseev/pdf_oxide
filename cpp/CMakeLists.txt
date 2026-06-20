cmake_minimum_required(VERSION 3.16)
project(pdf_oxide_cpp VERSION 0.3.68 LANGUAGES CXX)

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

# Header-only interface library consumers link against.
add_library(pdf_oxide_cpp INTERFACE)
target_include_directories(pdf_oxide_cpp INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${PDF_OXIDE_INCLUDE_DIR}")

find_library(PDF_OXIDE_NATIVE NAMES pdf_oxide libpdf_oxide PATHS "${PDF_OXIDE_LIB_DIR}" NO_DEFAULT_PATH)
if(NOT PDF_OXIDE_NATIVE)
  message(FATAL_ERROR "libpdf_oxide not found in ${PDF_OXIDE_LIB_DIR}; set -DPDF_OXIDE_LIB_DIR=...")
endif()
target_link_libraries(pdf_oxide_cpp INTERFACE "${PDF_OXIDE_NATIVE}")

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
