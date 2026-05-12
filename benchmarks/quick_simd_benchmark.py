#!/usr/bin/env python3
"""Quick SIMD comparison benchmark for arbitrage update APIs."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from negcycle._negcycle_native import available_backends


BACKEND_MODULES = {
    "generic": "negcycle._generic",
    "x86_sse": "negcycle._sse",
    "x86_avx": "negcycle._avx",
    "x86_avx2": "negcycle._avx2",
    "x86_avx512": "negcycle._avx512",
    "macos_arm64_neon": "negcycle._macos_arm64_neon",
    "linux_loongarch64_lsx": "negcycle._linux_loongarch64_lsx",
    "linux_loongarch64_lasx": "negcycle._linux_loongarch64_lasx",
}

UPDATE_FUNCTIONS = (
    "add_quote_and_find_best_arbitrage",
    "add_quote_and_find_arbitrage",
    "add_book_and_find_best_arbitrage",
    "add_book_and_find_arbitrage",
)
FUNCTION_VARIANTS = {
    "add_quote_and_find_best_arbitrage": "best",
    "add_quote_and_find_arbitrage": "all",
    "add_book_and_find_best_arbitrage": "best",
    "add_book_and_find_arbitrage": "all",
}
VARIANT_PASS_MULTIPLIERS = {
    "best": 10.0,
    "all": 0.5,
}


@dataclass(frozen=True)
class QuoteRow:
    base: str
    quote: str
    bid: float
    ask: float


@dataclass(frozen=True)
class Result:
    backend: str
    function: str
    calls: int
    seconds: float
    calls_per_second: float
    mean_us: float
    speedup_vs_generic: float | None
    returned: int


def parse_pair(ticker: str) -> tuple[str, str]:
    if ":" in ticker:
        ticker = ticker.split(":", 1)[1]
    if "-" in ticker:
        base, quote = ticker.split("-", 1)
    elif "/" in ticker:
        base, quote = ticker.split("/", 1)
    else:
        raise ValueError(f"unsupported ticker format: {ticker!r}")
    if not base or not quote:
        raise ValueError(f"unsupported ticker format: {ticker!r}")
    return base, quote


def load_rows(path: Path, limit: int | None) -> list[QuoteRow]:
    rows: list[QuoteRow] = []
    with path.expanduser().open(newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            base, quote = parse_pair(record["ticker"])
            bid = float(record["bid_price"])
            ask = float(record["ask_price"])
            if bid <= 0.0 or ask <= 0.0 or bid > ask:
                continue
            rows.append(QuoteRow(base=base, quote=quote, bid=bid, ask=ask))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no valid quote rows loaded from {path}")
    return rows


def backend_availability() -> dict[str, tuple[bool, bool]]:
    return {record.name: (record.compiled, record.available) for record in available_backends()}


def selected_backends(requested: list[str] | None, force_unavailable: bool) -> list[str]:
    availability = backend_availability()
    names = requested if requested else list(BACKEND_MODULES)
    selected: list[str] = []
    for name in names:
        compiled, available = availability.get(name, (False, False))
        if force_unavailable or (compiled and available):
            selected.append(name)
        else:
            print(
                f"skipping {name}: compiled={compiled} available={available}",
                file=sys.stderr,
                flush=True,
            )
    return selected


def load_detector(backend: str):
    return importlib.import_module(BACKEND_MODULES[backend]).ArbitrageDetector


def count_returned(value) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    return 1


def update_factor(index: int) -> float:
    return 1.0 + (1.0e-7 if index % 2 == 0 else -1.0e-7)


def make_handler(
    detector,
    function: str,
    *,
    fee_bps: float,
    max_cycle_length: int,
) -> Callable[[int, QuoteRow], object]:
    method = getattr(detector, function)
    if function.startswith("add_quote"):
        return lambda index, row: method(
            row.base,
            row.quote,
            row.bid * update_factor(index),
            fee_bps,
            max_cycle_length,
        )
    return lambda index, row: method(
        row.base,
        row.quote,
        row.bid * update_factor(index),
        row.ask * update_factor(index),
        fee_bps,
        max_cycle_length,
    )


def passes_for_function(base_passes: int, function: str) -> int:
    variant = FUNCTION_VARIANTS[function]
    return max(1, round(base_passes * VARIANT_PASS_MULTIPLIERS[variant]))


def run_one(
    *,
    backend: str,
    function: str,
    rows: list[QuoteRow],
    passes: int,
    fee_bps: float,
    max_cycle_length: int,
    generic_rate: float | None,
) -> Result:
    detector = load_detector(backend)()
    handler = make_handler(
        detector,
        function,
        fee_bps=fee_bps,
        max_cycle_length=max_cycle_length,
    )
    calls = passes * len(rows)
    returned = 0
    was_enabled = gc.isenabled()
    gc.disable()
    start = time.perf_counter()
    try:
        for pass_index in range(passes):
            offset = pass_index * len(rows)
            for row_index, row in enumerate(rows):
                returned = count_returned(handler(offset + row_index, row))
    finally:
        seconds = time.perf_counter() - start
        if was_enabled:
            gc.enable()
    calls_per_second = calls / seconds if seconds > 0 else 0.0
    speedup = None
    if generic_rate is not None and generic_rate > 0:
        speedup = calls_per_second / generic_rate
    return Result(
        backend=backend,
        function=function,
        calls=calls,
        seconds=seconds,
        calls_per_second=calls_per_second,
        mean_us=(seconds / calls * 1.0e6) if calls else 0.0,
        speedup_vs_generic=speedup,
        returned=returned,
    )


HEADERS = [
    "backend",
    "function",
    "calls",
    "seconds",
    "calls/s",
    "mean_us",
    "speedup",
    "returned",
]
WIDTHS = [16, 35, 10, 10, 12, 10, 8, 8]


def print_row(values: list[str]) -> None:
    print("  ".join(value[: WIDTHS[i]].ljust(WIDTHS[i]) for i, value in enumerate(values)), flush=True)


def print_result(result: Result) -> None:
    print_row(
        [
            result.backend,
            result.function,
            str(result.calls),
            f"{result.seconds:.4f}",
            f"{result.calls_per_second:.0f}",
            f"{result.mean_us:.3f}",
            "" if result.speedup_vs_generic is None else f"{result.speedup_vs_generic:.2f}x",
            str(result.returned),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick SIMD benchmark using a compact Massive currency quote CSV."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path.home() / "benchmark_data.csv",
        help="Input CSV path. Defaults to ~/benchmark_data.csv.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=4,
        help="Replay passes over the in-memory row list. Defaults to 4.",
    )
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--max-cycle-length", type=int, default=3)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument(
        "--backend",
        action="append",
        choices=sorted(BACKEND_MODULES),
        help="Backend to benchmark. May be repeated. Defaults to all runnable backends.",
    )
    parser.add_argument(
        "--function",
        action="append",
        choices=UPDATE_FUNCTIONS,
        help="Function to benchmark. May be repeated. Defaults to all update functions.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=("best", "all"),
        help="Benchmark only best-cycle or all-cycle variants. May be repeated.",
    )
    parser.add_argument(
        "--force-unavailable",
        action="store_true",
        help="Run requested backends even if runtime detection says unavailable.",
    )
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes must be positive")
    if args.row_limit is not None and args.row_limit < 1:
        parser.error("--row-limit must be positive")
    if args.max_cycle_length < 3:
        parser.error("--max-cycle-length must be at least 3")
    return args


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input, args.row_limit)
    backends = selected_backends(args.backend, args.force_unavailable)
    if not backends:
        raise SystemExit("no runnable backends selected")
    if "generic" in backends:
        backends = ["generic"] + [backend for backend in backends if backend != "generic"]
    variants = set(args.variant or ("best", "all"))
    candidates = tuple(args.function) if args.function else UPDATE_FUNCTIONS
    functions = tuple(
        function for function in candidates if FUNCTION_VARIANTS[function] in variants
    )
    if not functions:
        raise SystemExit("no functions selected")

    print_row(HEADERS)
    print_row(["-" * width for width in WIDTHS])

    generic_rates: dict[str, float] = {}
    for backend in backends:
        for function in functions:
            result = run_one(
                backend=backend,
                function=function,
                rows=rows,
                passes=passes_for_function(args.passes, function),
                fee_bps=args.fee_bps,
                max_cycle_length=args.max_cycle_length,
                generic_rate=generic_rates.get(function),
            )
            if backend == "generic":
                generic_rates[function] = result.calls_per_second
            print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
