#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "common/backend.h"


namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_negcycle_native, m) {
    m.doc() = "negcycle native scaffold";

    nb::enum_<negcycle::BackendKind>(m, "BackendKind")
        .value("generic", negcycle::BackendKind::generic)
        .value("x86_sse", negcycle::BackendKind::x86_sse)
        .value("x86_avx", negcycle::BackendKind::x86_avx)
        .value("x86_avx2", negcycle::BackendKind::x86_avx2)
        .value("x86_avx512", negcycle::BackendKind::x86_avx512)
        .value("linux_aarch64_asimd", negcycle::BackendKind::linux_aarch64_asimd)
        .value("linux_aarch64_sve", negcycle::BackendKind::linux_aarch64_sve)
        .value("linux_aarch64_sve2", negcycle::BackendKind::linux_aarch64_sve2)
        .value("macos_arm64_neon", negcycle::BackendKind::macos_arm64_neon)
        .value("linux_loongarch64_lsx", negcycle::BackendKind::linux_loongarch64_lsx)
        .value("linux_loongarch64_lasx", negcycle::BackendKind::linux_loongarch64_lasx)
        .value("linux_powerpc64_vsx", negcycle::BackendKind::linux_powerpc64_vsx)
        .value("linux_riscv64_rvv", negcycle::BackendKind::linux_riscv64_rvv);

    nb::class_<negcycle::BackendRecord>(m, "BackendRecord")
        .def_rw("kind", &negcycle::BackendRecord::kind)
        .def_prop_ro("name", [](const negcycle::BackendRecord& r) { return std::string(r.name); })
        .def_rw("compiled", &negcycle::BackendRecord::compiled)
        .def_rw("available", &negcycle::BackendRecord::available);

    m.def("detect_best_backend", &negcycle::detect_best_backend,
          "Return the highest-priority backend available on this machine.");
    m.def("available_backends", &negcycle::available_backends,
          "Return compiled backends and whether they are currently usable.");

}
