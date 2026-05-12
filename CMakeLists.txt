cmake_minimum_required(VERSION 3.26)
project(negcycle LANGUAGES CXX)

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
    src/cpp/backends/x86_avx2.cpp
    src/cpp/backends/x86_avx512.cpp
  )
  list(APPEND STRIDE_ALIGN_COMPILE_DEFS
    STRIDE_ALIGN_HAVE_X86_SSE=1
    STRIDE_ALIGN_HAVE_X86_AVX=1
    STRIDE_ALIGN_HAVE_X86_AVX2=1
    STRIDE_ALIGN_HAVE_X86_AVX512=1
  )
elseif (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  list(APPEND STRIDE_ALIGN_BACKEND_SOURCES
    src/cpp/backends/macos_arm64_neon.cpp
  )
  list(APPEND STRIDE_ALIGN_COMPILE_DEFS
    STRIDE_ALIGN_HAVE_MACOS_ARM64_NEON=1
  )
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  list(APPEND STRIDE_ALIGN_BACKEND_SOURCES
    src/cpp/backends/linux_aarch64_asimd.cpp
    src/cpp/backends/linux_aarch64_sve.cpp
    src/cpp/backends/linux_aarch64_sve2.cpp
  )
  list(APPEND STRIDE_ALIGN_COMPILE_DEFS
    STRIDE_ALIGN_HAVE_LINUX_AARCH64_ASIMD=1
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
  _negcycle_native
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

if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND (STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "x86_64|amd64"))
  nanobind_add_module(
    _avx
    src/cpp/common/detector_base.cpp
    src/cpp/backends/x86_avx.cpp
  )

  nanobind_add_module(
    _avx2
    src/cpp/common/detector_base.cpp
    src/cpp/backends/x86_avx2.cpp
  )

  nanobind_add_module(
    _avx512
    src/cpp/common/detector_base.cpp
    src/cpp/backends/x86_avx512.cpp
  )

  nanobind_add_module(
    _sse
    src/cpp/common/detector_base.cpp
    src/cpp/backends/x86_sse.cpp
  )
endif()

if (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  nanobind_add_module(
    _macos_arm64_neon
    src/cpp/common/detector_base.cpp
    src/cpp/backends/macos_arm64_neon.cpp
  )
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  nanobind_add_module(
    _linux_aarch64_asimd
    src/cpp/common/detector_base.cpp
    src/cpp/backends/linux_aarch64_asimd.cpp
  )

  nanobind_add_module(
    _linux_aarch64_sve
    src/cpp/common/detector_base.cpp
    src/cpp/backends/linux_aarch64_sve.cpp
  )

  nanobind_add_module(
    _linux_aarch64_sve2
    src/cpp/common/detector_base.cpp
    src/cpp/backends/linux_aarch64_sve2.cpp
  )
endif()


if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "loongarch64")
  nanobind_add_module(
    _linux_loongarch64_lsx
    src/cpp/common/detector_base.cpp
    src/cpp/backends/linux_loongarch64_lsx.cpp
  )

  nanobind_add_module(
    _linux_loongarch64_lasx
    src/cpp/common/detector_base.cpp
    src/cpp/backends/linux_loongarch64_lasx.cpp
  )
endif()


target_compile_definitions(_negcycle_native PRIVATE ${STRIDE_ALIGN_COMPILE_DEFS})
target_include_directories(_negcycle_native PRIVATE src/cpp)
target_include_directories(_common PRIVATE src/cpp)
target_include_directories(_generic PRIVATE src/cpp)
if (TARGET _avx)
  target_include_directories(_avx PRIVATE src/cpp)
endif()
if (TARGET _avx2)
  target_include_directories(_avx2 PRIVATE src/cpp)
endif()
if (TARGET _avx512)
  target_include_directories(_avx512 PRIVATE src/cpp)
endif()
if (TARGET _sse)
  target_include_directories(_sse PRIVATE src/cpp)
endif()
if (TARGET _macos_arm64_neon)
  target_include_directories(_macos_arm64_neon PRIVATE src/cpp)
endif()
if (TARGET _linux_aarch64_asimd)
  target_include_directories(_linux_aarch64_asimd PRIVATE src/cpp)
endif()
if (TARGET _linux_aarch64_sve)
  target_include_directories(_linux_aarch64_sve PRIVATE src/cpp)
endif()
if (TARGET _linux_aarch64_sve2)
  target_include_directories(_linux_aarch64_sve2 PRIVATE src/cpp)
endif()
if (TARGET _linux_loongarch64_lsx)
  target_include_directories(_linux_loongarch64_lsx PRIVATE src/cpp)
endif()
if (TARGET _linux_loongarch64_lasx)
  target_include_directories(_linux_loongarch64_lasx PRIVATE src/cpp)
endif()

# x86 per-file ISA levels. Keep this here so real intrinsics can be dropped in later.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND (STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "x86_64|amd64"))
  set_source_files_properties(src/cpp/backends/x86_sse.cpp PROPERTIES COMPILE_OPTIONS "-msse2")
  set_source_files_properties(src/cpp/backends/x86_avx.cpp PROPERTIES COMPILE_OPTIONS "-mavx;-mtune=core-avx-i")
  set_source_files_properties(src/cpp/backends/x86_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
  set_source_files_properties(src/cpp/backends/x86_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f")
endif()

# Linux AArch64 ISA levels. These are intended to cover AWS Graviton families:
# baseline ASIMD/NEON for Graviton2, SVE for Graviton3, and SVE2 where the CPU
# and kernel expose it. Generic/import support stays at the baseline ISA.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  target_compile_options(_negcycle_native PRIVATE -march=armv8-a)
  target_compile_options(_common PRIVATE -march=armv8-a)
  target_compile_options(_generic PRIVATE -march=armv8-a)
  target_compile_options(_linux_aarch64_asimd PRIVATE -march=armv8-a+simd)
  target_compile_options(_linux_aarch64_sve PRIVATE -march=armv8.2-a+sve)
  target_compile_options(_linux_aarch64_sve2 PRIVATE -march=armv8.5-a+sve2)
endif()

# LoongArch target ISA levels. Keep the generic/import support modules scalar,
# compile the LSX module with LSX but not LASX, and reserve LASX codegen for the
# LASX module so import-time dispatch controls which instructions can execute.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND STRIDE_ALIGN_SYSTEM_PROCESSOR MATCHES "loongarch64")
  target_compile_options(_negcycle_native PRIVATE -march=loongarch64 -mno-lsx -mno-lasx)
  target_compile_options(_common PRIVATE -march=loongarch64 -mno-lsx -mno-lasx)
  target_compile_options(_generic PRIVATE -march=loongarch64 -mno-lsx -mno-lasx)
  target_compile_options(_linux_loongarch64_lsx PRIVATE -march=loongarch64 -mlsx -mno-lasx)
  target_compile_options(_linux_loongarch64_lasx PRIVATE -march=loongarch64 -mlsx -mlasx)

  foreach(target _negcycle_native _common _generic _linux_loongarch64_lsx _linux_loongarch64_lasx)
    target_link_options(${target} PRIVATE -static-libstdc++ -static-libgcc)
  endforeach()
endif()

set(STRIDE_ALIGN_INSTALL_TARGETS
  _negcycle_native
  _common
  _generic
)

if (TARGET _avx)
  list(APPEND STRIDE_ALIGN_INSTALL_TARGETS _avx)
endif()
if (TARGET _avx2)
  list(APPEND STRIDE_ALIGN_INSTALL_TARGETS _avx2)
endif()
if (TARGET _avx512)
  list(APPEND STRIDE_ALIGN_INSTALL_TARGETS _avx512)
endif()
if (TARGET _sse)
  list(APPEND STRIDE_ALIGN_INSTALL_TARGETS _sse)
endif()
if (TARGET _macos_arm64_neon)
  list(APPEND STRIDE_ALIGN_INSTALL_TARGETS _macos_arm64_neon)
endif()
if (TARGET _linux_aarch64_asimd)
  list(APPEND STRIDE_ALIGN_INSTALL_TARGETS _linux_aarch64_asimd)
endif()
if (TARGET _linux_aarch64_sve)
  list(APPEND STRIDE_ALIGN_INSTALL_TARGETS _linux_aarch64_sve)
endif()
if (TARGET _linux_aarch64_sve2)
  list(APPEND STRIDE_ALIGN_INSTALL_TARGETS _linux_aarch64_sve2)
endif()
if (TARGET _linux_loongarch64_lsx)
  list(APPEND STRIDE_ALIGN_INSTALL_TARGETS _linux_loongarch64_lsx)
endif()
if (TARGET _linux_loongarch64_lasx)
  list(APPEND STRIDE_ALIGN_INSTALL_TARGETS _linux_loongarch64_lasx)
endif()

install(TARGETS ${STRIDE_ALIGN_INSTALL_TARGETS} LIBRARY DESTINATION negcycle)
