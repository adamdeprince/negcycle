import importlib

import pytest

from negcycle import ArbitrageDetector, available_backends


BACKEND_MODULES = {
    "generic": "negcycle._generic",
    "x86_sse": "negcycle._sse",
    "x86_avx": "negcycle._avx",
    "x86_avx2": "negcycle._avx2",
    "x86_avx512": "negcycle._avx512",
    "macos_arm64_neon": "negcycle._macos_arm64_neon",
    "linux_aarch64_asimd": "negcycle._linux_aarch64_asimd",
    "linux_aarch64_sve": "negcycle._linux_aarch64_sve",
    "linux_loongarch64_lsx": "negcycle._linux_loongarch64_lsx",
    "linux_loongarch64_lasx": "negcycle._linux_loongarch64_lasx",
}


def cycle_names(cycle):
    return tuple(cycle.names)


def build_detector_with_class(detector_class):
    detector = detector_class()
    detector.add_quote("USD", "EUR", 1.2)
    detector.add_quote("EUR", "USD", 1.1)
    detector.add_quote("USD", "JPY", 1.1)
    detector.add_quote("JPY", "USD", 1.1)
    detector.add_quote("EUR", "JPY", 1.1)
    detector.add_quote("JPY", "EUR", 1.1)
    return detector


def build_detector():
    return build_detector_with_class(ArbitrageDetector)


def test_find_arbitrage_returns_all_negative_cycles_sorted_best_first():
    detector = build_detector()

    cycles = detector.find_arbitrage(3)

    assert [cycle_names(cycle) for cycle in cycles] == [
        ("USD", "EUR", "JPY", "USD"),
        ("USD", "JPY", "EUR", "USD"),
        ("USD", "EUR", "USD"),
        ("USD", "JPY", "USD"),
        ("EUR", "JPY", "EUR"),
    ]
    assert all(cycle.total_weight < 0.0 for cycle in cycles)
    assert [cycle.total_weight for cycle in cycles] == sorted(
        [cycle.total_weight for cycle in cycles]
    )

    best = detector.find_best_arbitrage(3)
    assert best is not None
    assert cycle_names(cycles[0]) == cycle_names(best)
    assert cycles[0].gain_factor == pytest.approx(best.gain_factor)


def test_update_methods_return_all_current_negative_cycles():
    detector = ArbitrageDetector()

    assert detector.add_quote_and_find_arbitrage("USD", "EUR", 1.2, 0.0, 3) == []
    assert detector.add_quote_and_find_arbitrage("EUR", "JPY", 1.2, 0.0, 3) == []

    cycles = detector.add_quote_and_find_arbitrage("JPY", "USD", 1.2, 0.0, 3)
    assert [cycle_names(cycle) for cycle in cycles] == [
        ("USD", "EUR", "JPY", "USD"),
    ]

    cycles = detector.add_book_and_find_arbitrage("GBP", "CHF", 0.95, 1.1, 0.0, 3)
    assert [cycle_names(cycle) for cycle in cycles] == [
        ("USD", "EUR", "JPY", "USD"),
    ]


def runnable_backend_names():
    return [
        record.name
        for record in available_backends()
        if record.compiled and record.available and record.name in BACKEND_MODULES
    ]


@pytest.mark.parametrize("backend", runnable_backend_names())
def test_direct_backend_matches_generic_for_core_api(backend):
    backend_cls = importlib.import_module(BACKEND_MODULES[backend]).ArbitrageDetector
    generic_cls = importlib.import_module(BACKEND_MODULES["generic"]).ArbitrageDetector

    backend_detector = build_detector_with_class(backend_cls)
    generic_detector = build_detector_with_class(generic_cls)

    backend_cycles = backend_detector.find_arbitrage(3)
    generic_cycles = generic_detector.find_arbitrage(3)
    assert [cycle_names(cycle) for cycle in backend_cycles] == [
        cycle_names(cycle) for cycle in generic_cycles
    ]

    backend_best = backend_detector.find_best_arbitrage(3)
    generic_best = generic_detector.find_best_arbitrage(3)
    assert backend_best is not None
    assert generic_best is not None
    assert cycle_names(backend_best) == cycle_names(generic_best)
    assert backend_best.gain_factor == pytest.approx(generic_best.gain_factor)
