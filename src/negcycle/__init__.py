"""Top-level Python import surface for negcycle."""

from importlib import import_module

from negcycle._negcycle_native import BackendKind, available_backends, detect_best_backend
from negcycle._common import Cycle, Edge


_BACKEND_MODULES = {
    BackendKind.generic: "negcycle._generic",
    BackendKind.x86_sse: "negcycle._sse",
    BackendKind.x86_avx: "negcycle._avx",
    BackendKind.x86_avx512: "negcycle._avx512",
    BackendKind.macos_arm64_neon: "negcycle._macos_arm64_neon",
    BackendKind.linux_loongarch64_lsx: "negcycle._linux_loongarch64_lsx",
    BackendKind.linux_loongarch64_lasx: "negcycle._linux_loongarch64_lasx",
}


def _load_detector_class():
    backend = detect_best_backend()
    module_name = _BACKEND_MODULES.get(backend, "negcycle._generic")

    try:
        module = import_module(module_name)
    except ImportError:
        module = import_module("negcycle._generic")

    return module.ArbitrageDetector


def _restore_detector(state):
    detector = ArbitrageDetector()
    detector.__setstate__(state)
    return detector


ArbitrageDetector = _load_detector_class()


__all__ = [
    "detect_best_backend",
    "available_backends",
    "Cycle",
    "Edge",
    "ArbitrageDetector",
]
