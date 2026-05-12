import pytest

from negcycle import ArbitrageDetector


def cycle_names(cycle):
    return tuple(cycle.names)


def build_detector():
    detector = ArbitrageDetector()
    detector.add_quote("USD", "EUR", 1.2)
    detector.add_quote("EUR", "USD", 1.1)
    detector.add_quote("USD", "JPY", 1.1)
    detector.add_quote("JPY", "USD", 1.1)
    detector.add_quote("EUR", "JPY", 1.1)
    detector.add_quote("JPY", "EUR", 1.1)
    return detector


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
