"""Top-level Python import surface for negcycle."""

from __future__ import annotations

from functools import lru_cache as _lru_cache
from importlib import import_module as _import_module

try:
    import goblin_cpu_features as _cpu
except ImportError:  # pragma: no cover - package metadata requires this dependency
    _cpu = None

from negcycle._common import Cycle, Edge
from negcycle._negcycle_native import BackendKind as _BackendKind
from negcycle._negcycle_native import compiled_backends as _compiled_backends


_BACKEND_MODULES = {
    _BackendKind.generic: "negcycle._generic",
    _BackendKind.x86_sse: "negcycle._sse",
    _BackendKind.x86_avx: "negcycle._avx",
    _BackendKind.x86_avx2: "negcycle._avx2",
    _BackendKind.x86_avx512: "negcycle._avx512",
    _BackendKind.macos_arm64_neon: "negcycle._macos_arm64_neon",
    _BackendKind.linux_aarch64_asimd: "negcycle._linux_aarch64_asimd",
    _BackendKind.linux_aarch64_sve: "negcycle._linux_aarch64_sve",
    _BackendKind.linux_loongarch64_lsx: "negcycle._linux_loongarch64_lsx",
    _BackendKind.linux_loongarch64_lasx: "negcycle._linux_loongarch64_lasx",
}

_BACKEND_FEATURES = {
    _BackendKind.x86_sse: frozenset({"X86_SSE2"}),
    _BackendKind.x86_avx: frozenset({"X86_AVX"}),
    _BackendKind.x86_avx2: frozenset({"X86_AVX2"}),
    _BackendKind.x86_avx512: frozenset({"X86_AVX512F"}),
    _BackendKind.linux_aarch64_asimd: frozenset({"ARM_ASIMD"}),
    _BackendKind.linux_aarch64_sve: frozenset({"ARM_SVE"}),
    _BackendKind.macos_arm64_neon: frozenset({"ARM_ASIMD"}),
    _BackendKind.linux_loongarch64_lsx: frozenset({"LA_LSX"}),
    _BackendKind.linux_loongarch64_lasx: frozenset({"LA_LASX"}),
    _BackendKind.linux_powerpc64_vsx: frozenset({"PPC_VSX"}),
    _BackendKind.linux_riscv64_rvv: frozenset({"RV_V"}),
}

_DEFAULT_BACKEND_ORDER = (
    _BackendKind.x86_avx512,
    _BackendKind.x86_avx2,
    _BackendKind.x86_avx,
    _BackendKind.x86_sse,
    _BackendKind.macos_arm64_neon,
    _BackendKind.linux_aarch64_sve,
    _BackendKind.linux_aarch64_asimd,
    _BackendKind.linux_loongarch64_lasx,
    _BackendKind.linux_loongarch64_lsx,
    _BackendKind.linux_powerpc64_vsx,
    _BackendKind.linux_riscv64_rvv,
    _BackendKind.generic,
)


@_lru_cache(maxsize=1)
def _detected_feature_names() -> frozenset[str]:
    if _cpu is None:
        return frozenset()
    return frozenset(feature.name for feature in _cpu.detect())


def backend_is_available(kind):
    """Return whether a compiled backend can run on this CPU."""
    if kind not in _BACKEND_MODULES:
        return False
    if kind == _BackendKind.generic:
        return True
    required = _BACKEND_FEATURES.get(kind)
    if required is None:
        return False
    return required.issubset(_detected_feature_names())


def available_backends():
    """Return compiled backends with availability filled from CPU features."""
    records = list(_compiled_backends())
    for record in records:
        record.available = bool(record.compiled and backend_is_available(record.kind))
    return records


def detect_best_backend():
    """Return the default backend selected for this CPU."""
    available = {record.kind for record in available_backends() if record.available}
    for kind in _DEFAULT_BACKEND_ORDER:
        if kind in available:
            return kind
    return _BackendKind.generic


def _load_detector_class():
    backend = detect_best_backend()
    module_name = _BACKEND_MODULES.get(backend, "negcycle._generic")

    try:
        module = _import_module(module_name)
    except ImportError:
        module = _import_module("negcycle._generic")

    return module.ArbitrageDetector


def _restore_detector(state):
    detector = ArbitrageDetector()
    detector.__setstate__(state)
    return detector


ArbitrageDetector = _load_detector_class()


__all__ = [
    "ArbitrageDetector",
    "Cycle",
    "Edge",
    "available_backends",
    "backend_is_available",
    "detect_best_backend",
]
