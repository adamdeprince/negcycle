#pragma once

#include <string_view>
#include <vector>

namespace arbcycle {

enum class BackendKind {
    generic = 0,
    x86_sse,
    x86_avx,
    x86_avx512,
    linux_aarch64_asimd,
    linux_aarch64_neon,
    linux_aarch64_sve,
    linux_aarch64_sve2,
    macos_arm64_neon,
    linux_loongarch64_lsx,
    linux_loongarch64_lasx,
    linux_powerpc64_vsx,
    linux_riscv64_rvv,
};

struct BackendRecord {
    BackendKind kind;
    const char* name;
    bool compiled;
    bool available;
};

constexpr std::string_view backend_kind_name(BackendKind kind) noexcept {
    switch (kind) {
        case BackendKind::generic: return "generic";
        case BackendKind::x86_sse: return "x86_sse";
        case BackendKind::x86_avx: return "x86_avx";
        case BackendKind::x86_avx512: return "x86_avx512";
        case BackendKind::linux_aarch64_asimd: return "linux_aarch64_asimd";
        case BackendKind::linux_aarch64_neon: return "linux_aarch64_neon";
        case BackendKind::linux_aarch64_sve: return "linux_aarch64_sve";
        case BackendKind::linux_aarch64_sve2: return "linux_aarch64_sve2";
        case BackendKind::macos_arm64_neon: return "macos_arm64_neon";
        case BackendKind::linux_loongarch64_lsx: return "linux_loongarch64_lsx";
        case BackendKind::linux_loongarch64_lasx: return "linux_loongarch64_lasx";
        case BackendKind::linux_powerpc64_vsx: return "linux_powerpc64_vsx";
        case BackendKind::linux_riscv64_rvv: return "linux_riscv64_rvv";
    }
    return "unknown";
}

BackendKind detect_best_backend() noexcept;
bool backend_is_available(BackendKind kind) noexcept;
std::vector<BackendRecord> available_backends();

} // namespace arbcycle
