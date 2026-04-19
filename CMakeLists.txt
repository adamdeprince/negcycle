cmake_minimum_required(VERSION 3.26)
project(arbcycle LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING
    "This project is intended to be built through scikit-build-core.
"
    "Typical commands:
"
    "  pip install .
"
    "  pip install --no-build-isolation -ve .
"
  )
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Normalize CMake's per-config defaults so every build type uses -O3 and does
# not add debug-symbol or no-optimization flags behind our back.
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_CXX_FLAGS_${config} "-O3")
endforeach()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

set(STRIDE_ALIGN_COMMON_SOURCES
  src/cpp/module.cpp
  src/cpp/detect/detect.cpp
)

set(STRIDE_ALIGN_BACKEND_SOURCES
  src/cpp/backends/generic.cpp
)

set(STRIDE_ALIGN_COMPILE_DEFS
  STRIDE_ALIGN_HAVE_GENERIC=1
)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" STRIDE_ALIGN_SYSTEM_PROCESSOR)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND (STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "x86_64|amd64"))
  list(APPEND STRIDE_ALIGN_BACKEND_SOURCES
    src/cpp/backends/x86_sse.cpp
    src/cpp/backends/x86_avx.cpp
  )
  list(APPEND STRIDE_ALIGN_COMPILE_DEFS
    STRIDE_ALIGN_HAVE_X86_SSE=1
    STRIDE_ALIGN_HAVE_X86_AVX=1
  )
elseif (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  list(APPEND STRIDE_ALIGN_BACKEND_SOURCES
    src/cpp/backends/macos_arm64_neon.cpp
    src/cpp/backends/macos_arm64_sme.cpp
  )
  list(APPEND STRIDE_ALIGN_COMPILE_DEFS
    STRIDE_ALIGN_HAVE_MACOS_ARM64_NEON=1
    STRIDE_ALIGN_HAVE_MACOS_ARM64_SME=1
  )
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  list(APPEND STRIDE_ALIGN_BACKEND_SOURCES
    src/cpp/backends/linux_aarch64_asimd.cpp
    src/cpp/backends/linux_aarch64_neon.cpp
    src/cpp/backends/linux_aarch64_sve.cpp
    src/cpp/backends/linux_aarch64_sve2.cpp
  )
  list(APPEND STRIDE_ALIGN_COMPILE_DEFS
    STRIDE_ALIGN_HAVE_LINUX_AARCH64_ASIMD=1
    STRIDE_ALIGN_HAVE_LINUX_AARCH64_NEON=1
    STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE=1
    STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE2=1
  )
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "loongarch64")
  list(APPEND STRIDE_ALIGN_BACKEND_SOURCES
    src/cpp/backends/linux_loongarch64_lsx.cpp
    src/cpp/backends/linux_loongarch64_lasx.cpp
  )
  list(APPEND STRIDE_ALIGN_COMPILE_DEFS
    STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LSX=1
    STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LASX=1
  )
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc64")
  list(APPEND STRIDE_ALIGN_BACKEND_SOURCES
    src/cpp/backends/linux_powerpc64_vsx.cpp
  )
  list(APPEND STRIDE_ALIGN_COMPILE_DEFS
    STRIDE_ALIGN_HAVE_LINUX_POWERPC64_VSX=1
  )
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "riscv64")
  list(APPEND STRIDE_ALIGN_BACKEND_SOURCES
    src/cpp/backends/linux_riscv64_rvv.cpp
  )
  list(APPEND STRIDE_ALIGN_COMPILE_DEFS
    STRIDE_ALIGN_HAVE_LINUX_RISCV64_RVV=1
  )
endif()

nanobind_add_module(
  _arbcycle_native
  ${STRIDE_ALIGN_COMMON_SOURCES}
)

nanobind_add_module(
  _common
  src/cpp/common/types_module.cpp
)

nanobind_add_module(
  _generic
  src/cpp/common/detector_base.cpp
  src/cpp/backends/generic.cpp
)

nanobind_add_module(
  _avx
  src/cpp/common/detector_base.cpp
  src/cpp/backends/x86_avx.cpp
)

nanobind_add_module(
  _sse
  src/cpp/common/detector_base.cpp
  src/cpp/backends/x86_sse.cpp
)


target_compile_definitions(_arbcycle_native PRIVATE ${STRIDE_ALIGN_COMPILE_DEFS})
target_include_directories(_arbcycle_native PRIVATE src/cpp)
target_include_directories(_common PRIVATE src/cpp)
target_include_directories(_generic PRIVATE src/cpp)
target_include_directories(_avx PRIVATE src/cpp)
target_include_directories(_sse PRIVATE src/cpp)

# x86 per-file ISA levels. Keep this here so real intrinsics can be dropped in later.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND (STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "x86_64|amd64"))
  set_source_files_properties(src/cpp/backends/x86_sse.cpp PROPERTIES COMPILE_OPTIONS "-msse2")
  set_source_files_properties(src/cpp/backends/x86_avx.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
endif()

install(TARGETS _arbcycle_native _common _generic _avx _sse LIBRARY DESTINATION arbcycle)
