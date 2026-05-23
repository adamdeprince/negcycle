#include "common/backend.h"

#include <array>

namespace negcycle {
namespace {

constexpr std::array<BackendKind, 13> kAllBackends = {
    BackendKind::generic,
    BackendKind::x86_sse,
    BackendKind::x86_avx,
    BackendKind::x86_avx2,
    BackendKind::x86_avx512,
    BackendKind::linux_aarch64_asimd,
    BackendKind::linux_aarch64_sve,
    BackendKind::linux_aarch64_sve2,
    BackendKind::macos_arm64_neon,
    BackendKind::linux_loongarch64_lsx,
    BackendKind::linux_loongarch64_lasx,
    BackendKind::linux_powerpc64_vsx,
    BackendKind::linux_riscv64_rvv,
};

} // namespace

bool backend_is_compiled(BackendKind kind) noexcept {
    switch (kind) {
        case BackendKind::generic:
#ifdef STRIDE_ALIGN_HAVE_GENERIC
            return true;
#else
            return false;
#endif
        case BackendKind::x86_sse:
#ifdef STRIDE_ALIGN_HAVE_X86_SSE
            return true;
#else
            return false;
#endif
        case BackendKind::x86_avx:
#ifdef STRIDE_ALIGN_HAVE_X86_AVX
            return true;
#else
            return false;
#endif
        case BackendKind::x86_avx2:
#ifdef STRIDE_ALIGN_HAVE_X86_AVX2
            return true;
#else
            return false;
#endif
        case BackendKind::x86_avx512:
#ifdef STRIDE_ALIGN_HAVE_X86_AVX512
            return true;
#else
            return false;
#endif
        case BackendKind::linux_aarch64_asimd:
#ifdef STRIDE_ALIGN_HAVE_LINUX_AARCH64_ASIMD
            return true;
#else
            return false;
#endif
        case BackendKind::linux_aarch64_sve:
#ifdef STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE
            return true;
#else
            return false;
#endif
        case BackendKind::linux_aarch64_sve2:
#ifdef STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE2
            return true;
#else
            return false;
#endif
        case BackendKind::macos_arm64_neon:
#ifdef STRIDE_ALIGN_HAVE_MACOS_ARM64_NEON
            return true;
#else
            return false;
#endif
        case BackendKind::linux_loongarch64_lsx:
#ifdef STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LSX
            return true;
#else
            return false;
#endif
        case BackendKind::linux_loongarch64_lasx:
#ifdef STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LASX
            return true;
#else
            return false;
#endif
        case BackendKind::linux_powerpc64_vsx:
#ifdef STRIDE_ALIGN_HAVE_LINUX_POWERPC64_VSX
            return true;
#else
            return false;
#endif
        case BackendKind::linux_riscv64_rvv:
#ifdef STRIDE_ALIGN_HAVE_LINUX_RISCV64_RVV
            return true;
#else
            return false;
#endif
    }
    return false;
}

std::vector<BackendRecord> compiled_backends() {
    std::vector<BackendRecord> out;
    out.reserve(kAllBackends.size());

    for (const auto kind : kAllBackends) {
        if (!backend_is_compiled(kind)) {
            continue;
        }

        out.push_back(BackendRecord{
            .kind = kind,
            .name = backend_kind_name(kind).data(),
            .compiled = true,
            .available = false,
        });
    }

    return out;
}

} // namespace negcycle
