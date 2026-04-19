#include "common/backend.h"

#include <array>
#include <cstdint>

#if defined(__linux__)
  #include <sys/auxv.h>
  #include <unistd.h>
#endif

#if defined(__APPLE__)
  #include <sys/sysctl.h>
#endif

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
  #if defined(__x86_64__) || defined(__i386__)
    #include <cpuid.h>
  #endif
#endif

#if defined(__linux__) && defined(__aarch64__)
  #include <asm/hwcap.h>
#endif

#if defined(__linux__) && defined(__loongarch64)
  #include <asm/hwcap.h>
#endif

#if defined(__linux__) && (defined(__powerpc64__) || defined(__ppc64__))
  #include <asm/cputable.h>
#endif

#if defined(__linux__) && defined(__riscv)
  #include <asm/hwprobe.h>
  #include <sys/syscall.h>
#endif

namespace arbcycle {
namespace {

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
struct CpuidRegs {
    std::uint32_t eax{};
    std::uint32_t ebx{};
    std::uint32_t ecx{};
    std::uint32_t edx{};
};

CpuidRegs cpuid(std::uint32_t leaf, std::uint32_t subleaf = 0) noexcept {
    CpuidRegs r{};
#if defined(_MSC_VER)
    int regs[4]{};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<std::uint32_t>(regs[0]);
    r.ebx = static_cast<std::uint32_t>(regs[1]);
    r.ecx = static_cast<std::uint32_t>(regs[2]);
    r.edx = static_cast<std::uint32_t>(regs[3]);
#elif defined(__GNUC__) || defined(__clang__)
    std::uint32_t a{}, b{}, c{}, d{};
    __cpuid_count(leaf, subleaf, a, b, c, d);
    r.eax = a;
    r.ebx = b;
    r.ecx = c;
    r.edx = d;
#endif
    return r;
}

std::uint64_t xgetbv(std::uint32_t xcr) noexcept {
#if defined(_MSC_VER)
    return static_cast<std::uint64_t>(_xgetbv(xcr));
#elif defined(__GNUC__) || defined(__clang__)
    std::uint32_t eax = 0, edx = 0;
    __asm__ volatile(
        ".byte 0x0f, 0x01, 0xd0"
        : "=a"(eax), "=d"(edx)
        : "c"(xcr));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#else
    (void) xcr;
    return 0;
#endif
}

bool xcr0_has_bits(std::uint64_t mask) noexcept {
    return (xgetbv(0) & mask) == mask;
}

bool supports_x86_sse() noexcept {
    return true; // x86-64 baseline in this scaffold
}

bool supports_x86_avx() noexcept {
    const auto leaf0 = cpuid(0, 0);
    if (leaf0.eax < 1) {
        return false;
    }
    const auto leaf1 = cpuid(1, 0);
    constexpr std::uint32_t ECX_OSXSAVE = 1u << 27;
    constexpr std::uint32_t ECX_AVX = 1u << 28;
    const bool has_osxsave = (leaf1.ecx & ECX_OSXSAVE) != 0;
    const bool has_avx = (leaf1.ecx & ECX_AVX) != 0;
    return has_osxsave && has_avx && xcr0_has_bits((1ull << 1) | (1ull << 2));
}

bool supports_x86_avx512() noexcept {
    if (!supports_x86_avx()) {
        return false;
    }
    const auto leaf0 = cpuid(0, 0);
    if (leaf0.eax < 7) {
        return false;
    }
    const auto leaf7 = cpuid(7, 0);
    constexpr std::uint32_t EBX_AVX512F = 1u << 16;
    constexpr std::uint64_t XCR0_AVX512_MASK =
        (1ull << 1) | (1ull << 2) | (1ull << 5) | (1ull << 6) | (1ull << 7);
    return ((leaf7.ebx & EBX_AVX512F) != 0) && xcr0_has_bits(XCR0_AVX512_MASK);
}

#endif

#if defined(__linux__)
unsigned long linux_hwcap() noexcept {
    return getauxval(AT_HWCAP);
}

unsigned long linux_hwcap2() noexcept {
#ifdef AT_HWCAP2
    return getauxval(AT_HWCAP2);
#else
    return 0;
#endif
}
#endif

#if defined(__linux__) && defined(__aarch64__)
bool supports_linux_aarch64_asimd() noexcept {
#ifdef HWCAP_ASIMD
    return (linux_hwcap() & HWCAP_ASIMD) != 0;
#else
    return true;
#endif
}

bool supports_linux_aarch64_neon() noexcept {
    return supports_linux_aarch64_asimd();
}

bool supports_linux_aarch64_sve() noexcept {
#ifdef HWCAP_SVE
    return (linux_hwcap() & HWCAP_SVE) != 0;
#else
    return false;
#endif
}

bool supports_linux_aarch64_sve2() noexcept {
#ifdef HWCAP2_SVE2
    return (linux_hwcap2() & HWCAP2_SVE2) != 0;
#else
    return false;
#endif
}
#endif

#if defined(__APPLE__) && defined(__aarch64__)
bool apple_sysctl_flag(const char* name) noexcept {
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value != 0;
}

bool supports_macos_arm64_neon() noexcept {
    return true; // AdvSIMD/NEON is the baseline expectation for arm64 Apple builds here.
}

bool supports_macos_arm64_sme() noexcept {
    if (apple_sysctl_flag("hw.optional.arm.FEAT_SME")) {
        return true;
    }
#if defined(__ARM_FEATURE_SME)
    return true;
#else
    return false;
#endif
}
#endif

#if defined(__linux__) && defined(__loongarch64)
bool supports_linux_loongarch64_lsx() noexcept {
#ifdef HWCAP_LOONGARCH_LSX
    return (linux_hwcap() & HWCAP_LOONGARCH_LSX) != 0;
#else
    return false;
#endif
}

bool supports_linux_loongarch64_lasx() noexcept {
#ifdef HWCAP_LOONGARCH_LASX
    return (linux_hwcap() & HWCAP_LOONGARCH_LASX) != 0;
#else
    return false;
#endif
}
#endif

#if defined(__linux__) && (defined(__powerpc64__) || defined(__ppc64__))
bool supports_linux_powerpc64_vsx() noexcept {
#ifdef PPC_FEATURE_HAS_VSX
    return (linux_hwcap() & PPC_FEATURE_HAS_VSX) != 0;
#else
    return false;
#endif
}
#endif

#if defined(__linux__) && defined(__riscv)
bool supports_linux_riscv64_rvv() noexcept {
#if defined(SYS_riscv_hwprobe) && defined(RISCV_HWPROBE_KEY_IMA_EXT_0) && defined(RISCV_HWPROBE_IMA_V)
    riscv_hwprobe pair{};
    pair.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
    long rc = syscall(SYS_riscv_hwprobe, &pair, 1, 0, nullptr, 0);
    if (rc == 0) {
        return (static_cast<std::uint64_t>(pair.value) & RISCV_HWPROBE_IMA_V) != 0;
    }
#endif
#ifdef COMPAT_HWCAP_ISA_V
    return (linux_hwcap() & COMPAT_HWCAP_ISA_V) != 0;
#else
    return false;
#endif
}
#endif

} // namespace

bool backend_is_available(BackendKind kind) noexcept {
    switch (kind) {
        case BackendKind::generic:
            return true;
        case BackendKind::x86_sse:
#if defined(STRIDE_ALIGN_HAVE_X86_SSE)
            return supports_x86_sse();
#else
            return false;
#endif
        case BackendKind::x86_avx:
#if defined(STRIDE_ALIGN_HAVE_X86_AVX)
            return supports_x86_avx();
#else
            return false;
#endif
        case BackendKind::x86_avx512:
#if defined(STRIDE_ALIGN_HAVE_X86_AVX512)
            return supports_x86_avx512();
#else
            return false;
#endif
        case BackendKind::linux_aarch64_asimd:
#if defined(STRIDE_ALIGN_HAVE_LINUX_AARCH64_ASIMD)
            return supports_linux_aarch64_asimd();
#else
            return false;
#endif
        case BackendKind::linux_aarch64_neon:
#if defined(STRIDE_ALIGN_HAVE_LINUX_AARCH64_NEON)
            return supports_linux_aarch64_neon();
#else
            return false;
#endif
        case BackendKind::linux_aarch64_sve:
#if defined(STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE)
            return supports_linux_aarch64_sve();
#else
            return false;
#endif
        case BackendKind::linux_aarch64_sve2:
#if defined(STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE2)
            return supports_linux_aarch64_sve2();
#else
            return false;
#endif
        case BackendKind::macos_arm64_neon:
#if defined(STRIDE_ALIGN_HAVE_MACOS_ARM64_NEON)
            return supports_macos_arm64_neon();
#else
            return false;
#endif
        case BackendKind::macos_arm64_sme:
#if defined(STRIDE_ALIGN_HAVE_MACOS_ARM64_SME)
            return supports_macos_arm64_sme();
#else
            return false;
#endif
        case BackendKind::linux_loongarch64_lsx:
#if defined(STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LSX)
            return supports_linux_loongarch64_lsx();
#else
            return false;
#endif
        case BackendKind::linux_loongarch64_lasx:
#if defined(STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LASX)
            return supports_linux_loongarch64_lasx();
#else
            return false;
#endif
        case BackendKind::linux_powerpc64_vsx:
#if defined(STRIDE_ALIGN_HAVE_LINUX_POWERPC64_VSX)
            return supports_linux_powerpc64_vsx();
#else
            return false;
#endif
        case BackendKind::linux_riscv64_rvv:
#if defined(STRIDE_ALIGN_HAVE_LINUX_RISCV64_RVV)
            return supports_linux_riscv64_rvv();
#else
            return false;
#endif
    }
    return false;
}

BackendKind detect_best_backend() noexcept {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    if (backend_is_available(BackendKind::x86_avx512)) return BackendKind::x86_avx512;
    if (backend_is_available(BackendKind::x86_avx)) return BackendKind::x86_avx;
    if (backend_is_available(BackendKind::x86_sse)) return BackendKind::x86_sse;
    return BackendKind::generic;
#elif defined(__APPLE__) && defined(__aarch64__)
    if (backend_is_available(BackendKind::macos_arm64_sme)) return BackendKind::macos_arm64_sme;
    if (backend_is_available(BackendKind::macos_arm64_neon)) return BackendKind::macos_arm64_neon;
    return BackendKind::generic;
#elif defined(__linux__) && defined(__aarch64__)
    if (backend_is_available(BackendKind::linux_aarch64_sve2)) return BackendKind::linux_aarch64_sve2;
    if (backend_is_available(BackendKind::linux_aarch64_sve)) return BackendKind::linux_aarch64_sve;
    if (backend_is_available(BackendKind::linux_aarch64_asimd)) return BackendKind::linux_aarch64_asimd;
    return BackendKind::generic;
#elif defined(__linux__) && defined(__loongarch64)
    if (backend_is_available(BackendKind::linux_loongarch64_lasx)) return BackendKind::linux_loongarch64_lasx;
    if (backend_is_available(BackendKind::linux_loongarch64_lsx)) return BackendKind::linux_loongarch64_lsx;
    return BackendKind::generic;
#elif defined(__linux__) && (defined(__powerpc64__) || defined(__ppc64__))
    if (backend_is_available(BackendKind::linux_powerpc64_vsx)) return BackendKind::linux_powerpc64_vsx;
    return BackendKind::generic;
#elif defined(__linux__) && defined(__riscv)
    if (backend_is_available(BackendKind::linux_riscv64_rvv)) return BackendKind::linux_riscv64_rvv;
    return BackendKind::generic;
#else
    return BackendKind::generic;
#endif
}

std::vector<BackendRecord> available_backends() {
    const std::array<BackendKind, 14> all = {
        BackendKind::generic,
        BackendKind::x86_sse,
        BackendKind::x86_avx,
        BackendKind::x86_avx512,
        BackendKind::linux_aarch64_asimd,
        BackendKind::linux_aarch64_neon,
        BackendKind::linux_aarch64_sve,
        BackendKind::linux_aarch64_sve2,
        BackendKind::macos_arm64_neon,
        BackendKind::macos_arm64_sme,
        BackendKind::linux_loongarch64_lsx,
        BackendKind::linux_loongarch64_lasx,
        BackendKind::linux_powerpc64_vsx,
        BackendKind::linux_riscv64_rvv,
    };

    std::vector<BackendRecord> out;
    out.reserve(all.size());

    for (const auto kind : all) {
        bool compiled = false;
        switch (kind) {
            case BackendKind::generic:
#ifdef STRIDE_ALIGN_HAVE_GENERIC
                compiled = true;
#endif
                break;
            case BackendKind::x86_sse:
#ifdef STRIDE_ALIGN_HAVE_X86_SSE
                compiled = true;
#endif
                break;
            case BackendKind::x86_avx:
#ifdef STRIDE_ALIGN_HAVE_X86_AVX
                compiled = true;
#endif
                break;
            case BackendKind::x86_avx512:
#ifdef STRIDE_ALIGN_HAVE_X86_AVX512
                compiled = true;
#endif
                break;
            case BackendKind::linux_aarch64_asimd:
#ifdef STRIDE_ALIGN_HAVE_LINUX_AARCH64_ASIMD
                compiled = true;
#endif
                break;
            case BackendKind::linux_aarch64_neon:
#ifdef STRIDE_ALIGN_HAVE_LINUX_AARCH64_NEON
                compiled = true;
#endif
                break;
            case BackendKind::linux_aarch64_sve:
#ifdef STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE
                compiled = true;
#endif
                break;
            case BackendKind::linux_aarch64_sve2:
#ifdef STRIDE_ALIGN_HAVE_LINUX_AARCH64_SVE2
                compiled = true;
#endif
                break;
            case BackendKind::macos_arm64_neon:
#ifdef STRIDE_ALIGN_HAVE_MACOS_ARM64_NEON
                compiled = true;
#endif
                break;
            case BackendKind::macos_arm64_sme:
#ifdef STRIDE_ALIGN_HAVE_MACOS_ARM64_SME
                compiled = true;
#endif
                break;
            case BackendKind::linux_loongarch64_lsx:
#ifdef STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LSX
                compiled = true;
#endif
                break;
            case BackendKind::linux_loongarch64_lasx:
#ifdef STRIDE_ALIGN_HAVE_LINUX_LOONGARCH64_LASX
                compiled = true;
#endif
                break;
            case BackendKind::linux_powerpc64_vsx:
#ifdef STRIDE_ALIGN_HAVE_LINUX_POWERPC64_VSX
                compiled = true;
#endif
                break;
            case BackendKind::linux_riscv64_rvv:
#ifdef STRIDE_ALIGN_HAVE_LINUX_RISCV64_RVV
                compiled = true;
#endif
                break;
        }

        if (compiled) {
            out.push_back(BackendRecord{
                .kind = kind,
                .name = backend_kind_name(kind).data(),
                .compiled = true,
                .available = backend_is_available(kind),
            });
        }
    }

    return out;
}

} // namespace arbcycle
