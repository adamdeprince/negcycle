"""Top-level Python import surface for arbcycle."""

from importlib import import_module

from arbcycle._arbcycle_native import BackendKind, detect_best_backend
from arbcycle._common import Cycle, Edge


_BACKEND_MODULES = {
    BackendKind.generic: "arbcycle._generic",
    BackendKind.x86_sse: "arbcycle._sse",
    BackendKind.x86_avx: "arbcycle._avx",
    BackendKind.x86_avx512: "arbcycle._avx512",
    BackendKind.macos_arm64_neon: "arbcycle._macos_arm64_neon",
}


def _load_detector_class():
    backend = detect_best_backend()
    module_name = _BACKEND_MODULES.get(backend, "arbcycle._generic")

    try:
        module = import_module(module_name)
    except ImportError:
        module = import_module("arbcycle._generic")

    return module.ArbitrageDetector


ArbitrageDetector = _load_detector_class()


__all__ = ["detect_best_backend", "Cycle", "Edge", "ArbitrageDetector"]
