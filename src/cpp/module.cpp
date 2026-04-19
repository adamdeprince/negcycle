#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "common/backend.h"


namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_arbcycle_native, m) {
    m.doc() = "arbcycle native scaffold";

    nb::enum_<arbcycle::BackendKind>(m, "BackendKind")
        .value("generic", arbcycle::BackendKind::generic)
        .value("x86_sse", arbcycle::BackendKind::x86_sse)
        .value("x86_avx", arbcycle::BackendKind::x86_avx)
        .value("linux_aarch64_asimd", arbcycle::BackendKind::linux_aarch64_asimd)
        .value("linux_aarch64_neon", arbcycle::BackendKind::linux_aarch64_neon)
        .value("linux_aarch64_sve", arbcycle::BackendKind::linux_aarch64_sve)
        .value("linux_aarch64_sve2", arbcycle::BackendKind::linux_aarch64_sve2)
        .value("macos_arm64_neon", arbcycle::BackendKind::macos_arm64_neon)
        .value("macos_arm64_sme", arbcycle::BackendKind::macos_arm64_sme)
        .value("linux_loongarch64_lsx", arbcycle::BackendKind::linux_loongarch64_lsx)
        .value("linux_loongarch64_lasx", arbcycle::BackendKind::linux_loongarch64_lasx)
        .value("linux_powerpc64_vsx", arbcycle::BackendKind::linux_powerpc64_vsx)
        .value("linux_riscv64_rvv", arbcycle::BackendKind::linux_riscv64_rvv);

    nb::class_<arbcycle::BackendRecord>(m, "BackendRecord")
        .def_rw("kind", &arbcycle::BackendRecord::kind)
        .def_prop_ro("name", [](const arbcycle::BackendRecord& r) { return std::string(r.name); })
        .def_rw("compiled", &arbcycle::BackendRecord::compiled)
        .def_rw("available", &arbcycle::BackendRecord::available);

    m.def("detect_best_backend", &arbcycle::detect_best_backend,
          "Return the highest-priority backend available on this machine.");
    m.def("available_backends", &arbcycle::available_backends,
          "Return compiled backends and whether they are currently usable.");

}
