#!/usr/bin/env python3
"""Benchmark best-cycle and all-cycle arbitrage APIs by backend module."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from negcycle._negcycle_native import available_backends


BACKEND_MODULES = {
    "generic": "negcycle._generic",
    "x86_sse": "negcycle._sse",
    "x86_avx": "negcycle._avx",
    "x86_avx512": "negcycle._avx512",
    "macos_arm64_neon": "negcycle._macos_arm64_neon",
    "linux_loongarch64_lsx": "negcycle._linux_loongarch64_lsx",
    "linux_loongarch64_lasx": "negcycle._linux_loongarch64_lasx",
}

DEFAULT_DATA_PATH = Path(__file__).with_name("benchmark_data.csv")

SIMD_BACKENDS = {
    "x86_sse",
    "x86_avx",
    "x86_avx512",
    "macos_arm64_neon",
    "linux_loongarch64_lsx",
    "linux_loongarch64_lasx",
}


@dataclass(frozen=True)
class BookRow:
    base: str
    quote: str
    bid: float
    ask: float


@dataclass(frozen=True)
class CsvColumns:
    pair: str | None
    base: str | None
    quote: str | None
    bid: str
    ask: str


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    module: str
    function: str
    iterations: int
    warmups: int
    calls: int
    max_cycle_length: int
    currencies: int
    books: int
    total_seconds: float
    mean_us: float
    median_us: float
    min_us: float
    max_us: float
    returned: int


def choose_column(
    fieldnames: list[str],
    explicit: str | None,
    candidates: tuple[str, ...],
    what: str,
) -> str:
    if explicit is not None:
        if explicit not in fieldnames:
            raise ValueError(
                f"{what} column {explicit!r} is not present; available columns: {fieldnames}"
            )
        return explicit

    lower_to_name = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lower_to_name:
            return lower_to_name[candidate]

    raise ValueError(
        f"could not infer {what} column; pass the column name explicitly. "
        f"Available columns: {fieldnames}"
    )


def choose_optional_column(
    fieldnames: list[str],
    explicit: str | None,
    candidates: tuple[str, ...],
) -> str | None:
    if explicit is not None:
        if explicit not in fieldnames:
            raise ValueError(
                f"column {explicit!r} is not present; available columns: {fieldnames}"
            )
        return explicit

    lower_to_name = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lower_to_name:
            return lower_to_name[candidate]
    return None


def sniff_dialect(path: Path) -> csv.Dialect:
    with path.open(newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample)
    except csv.Error:
        return csv.excel


def infer_columns(
    fieldnames: list[str],
    *,
    pair_column: str | None,
    base_column: str | None,
    quote_column: str | None,
    bid_column: str | None,
    ask_column: str | None,
) -> CsvColumns:
    pair = choose_optional_column(
        fieldnames,
        pair_column,
        ("ticker", "pair", "symbol", "instrument", "currency_pair", "market"),
    )
    base = choose_optional_column(
        fieldnames,
        base_column,
        ("base", "base_currency", "from", "from_symbol", "from_code"),
    )
    quote = choose_optional_column(
        fieldnames,
        quote_column,
        ("quote", "quote_currency", "to", "to_symbol", "to_code"),
    )

    if pair is None and (base is None or quote is None):
        raise ValueError(
            "could not infer pair columns; pass either --pair-column or both "
            f"--base-column/--quote-column. Available columns: {fieldnames}"
        )

    return CsvColumns(
        pair=pair,
        base=base,
        quote=quote,
        bid=choose_column(fieldnames, bid_column, ("bid", "bid_price"), "bid"),
        ask=choose_column(fieldnames, ask_column, ("ask", "ask_price"), "ask"),
    )


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


def load_books(
    path: Path,
    *,
    pair_column: str | None,
    base_column: str | None,
    quote_column: str | None,
    bid_column: str | None,
    ask_column: str | None,
) -> list[BookRow]:
    rows: list[BookRow] = []
    dialect = sniff_dialect(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not have a CSV header")
        columns = infer_columns(
            reader.fieldnames,
            pair_column=pair_column,
            base_column=base_column,
            quote_column=quote_column,
            bid_column=bid_column,
            ask_column=ask_column,
        )

        for row in reader:
            if columns.pair is not None:
                base, quote = parse_pair(row[columns.pair])
            else:
                assert columns.base is not None
                assert columns.quote is not None
                base = row[columns.base]
                quote = row[columns.quote]
            bid = float(row[columns.bid])
            ask = float(row[columns.ask])
            rows.append(BookRow(base=base, quote=quote, bid=bid, ask=ask))

    if not rows:
        raise ValueError(f"no book rows loaded from {path}")
    return rows


def backend_availability() -> dict[str, tuple[bool, bool]]:
    return {record.name: (record.compiled, record.available) for record in available_backends()}


def available_backend_modules(include_generic: bool) -> list[tuple[str, str]]:
    wanted = set(SIMD_BACKENDS)
    if include_generic:
        wanted.add("generic")

    modules: list[tuple[str, str]] = []
    for record in available_backends():
        if not record.compiled or not record.available:
            continue
        if record.name not in wanted:
            continue
        module_name = BACKEND_MODULES.get(record.name)
        if module_name is None:
            continue
        modules.append((record.name, module_name))

    modules.sort(key=lambda item: (item[0] != "generic", item[0]))
    return modules


def load_backend_detector(module_name: str):
    module = importlib.import_module(module_name)
    return module.ArbitrageDetector


def populate_detector(detector, books: Iterable[BookRow], fee_bps: float) -> None:
    for row in books:
        detector.add_book(row.base, row.quote, row.bid, row.ask, fee_bps)


def build_detector(detector_type, books: list[BookRow], fee_bps: float):
    detector = detector_type()
    populate_detector(detector, books, fee_bps)
    return detector


def count_returned(value) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    return 1


def update_factor(index: int) -> float:
    return 1.0 + (1.0e-6 if index % 2 == 0 else -1.0e-6)


def time_calls(
    call: Callable[[int], object],
    iterations: int,
    warmups: int,
) -> tuple[list[int], int]:
    for i in range(warmups):
        call(i)

    timings: list[int] = []
    last_count = 0
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(iterations):
            start = time.perf_counter_ns()
            value = call(i)
            stop = time.perf_counter_ns()
            timings.append(stop - start)
            last_count = count_returned(value)
    finally:
        if was_enabled:
            gc.enable()
    return timings, last_count


def time_row_calls(
    call: Callable[[int, BookRow], object],
    rows: list[BookRow],
    iterations: int,
    warmups: int,
) -> tuple[list[int], int]:
    for pass_index in range(warmups):
        for row_index, row in enumerate(rows):
            call(pass_index * len(rows) + row_index, row)

    timings: list[int] = []
    last_count = 0
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for pass_index in range(iterations):
            for row_index, row in enumerate(rows):
                index = pass_index * len(rows) + row_index
                start = time.perf_counter_ns()
                value = call(index, row)
                stop = time.perf_counter_ns()
                timings.append(stop - start)
                last_count = count_returned(value)
    finally:
        if was_enabled:
            gc.enable()
    return timings, last_count


def summarize(
    *,
    backend: str,
    module: str,
    function: str,
    iterations: int,
    warmups: int,
    calls: int,
    max_cycle_length: int,
    currencies: int,
    books: int,
    timings_ns: list[int],
    returned: int,
) -> BenchmarkResult:
    total_ns = sum(timings_ns)
    return BenchmarkResult(
        backend=backend,
        module=module,
        function=function,
        iterations=iterations,
        warmups=warmups,
        calls=calls,
        max_cycle_length=max_cycle_length,
        currencies=currencies,
        books=books,
        total_seconds=total_ns / 1.0e9,
        mean_us=statistics.fmean(timings_ns) / 1.0e3,
        median_us=statistics.median(timings_ns) / 1.0e3,
        min_us=min(timings_ns) / 1.0e3,
        max_us=max(timings_ns) / 1.0e3,
        returned=returned,
    )


def benchmark_backend(
    *,
    backend: str,
    module_name: str,
    books: list[BookRow],
    iterations: int,
    warmups: int,
    max_cycle_length: int,
    fee_bps: float,
) -> list[BenchmarkResult]:
    detector_type = load_backend_detector(module_name)
    currencies = len({row.base for row in books} | {row.quote for row in books})
    results: list[BenchmarkResult] = []

    def run(function: str, call: Callable[[int], object]) -> None:
        timings, returned = time_calls(call, iterations, warmups)
        results.append(
            summarize(
                backend=backend,
                module=module_name,
                function=function,
                iterations=iterations,
                warmups=warmups,
                calls=len(timings),
                max_cycle_length=max_cycle_length,
                currencies=currencies,
                books=len(books),
                timings_ns=timings,
                returned=returned,
            )
        )

    def run_rows(function: str, call: Callable[[int, BookRow], object]) -> None:
        timings, returned = time_row_calls(call, books, iterations, warmups)
        results.append(
            summarize(
                backend=backend,
                module=module_name,
                function=function,
                iterations=iterations,
                warmups=warmups,
                calls=len(timings),
                max_cycle_length=max_cycle_length,
                currencies=currencies,
                books=len(books),
                timings_ns=timings,
                returned=returned,
            )
        )

    detector = build_detector(detector_type, books, fee_bps)
    run("find_best_arbitrage", lambda _: detector.find_best_arbitrage(max_cycle_length))

    detector = build_detector(detector_type, books, fee_bps)
    run("find_arbitrage", lambda _: detector.find_arbitrage(max_cycle_length))

    detector = build_detector(detector_type, books, fee_bps)
    run_rows(
        "add_quote_and_find_best_arbitrage",
        lambda i, row: detector.add_quote_and_find_best_arbitrage(
            row.base,
            row.quote,
            row.bid * update_factor(i),
            fee_bps,
            max_cycle_length,
        ),
    )

    detector = build_detector(detector_type, books, fee_bps)
    run_rows(
        "add_quote_and_find_arbitrage",
        lambda i, row: detector.add_quote_and_find_arbitrage(
            row.base,
            row.quote,
            row.bid * update_factor(i),
            fee_bps,
            max_cycle_length,
        ),
    )

    detector = build_detector(detector_type, books, fee_bps)
    run_rows(
        "add_book_and_find_best_arbitrage",
        lambda i, row: detector.add_book_and_find_best_arbitrage(
            row.base,
            row.quote,
            row.bid * update_factor(i),
            row.ask * update_factor(i),
            fee_bps,
            max_cycle_length,
        ),
    )

    detector = build_detector(detector_type, books, fee_bps)
    run_rows(
        "add_book_and_find_arbitrage",
        lambda i, row: detector.add_book_and_find_arbitrage(
            row.base,
            row.quote,
            row.bid * update_factor(i),
            row.ask * update_factor(i),
            fee_bps,
            max_cycle_length,
        ),
    )

    return results


def write_csv(results: list[BenchmarkResult]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=list(BenchmarkResult.__annotations__))
    writer.writeheader()
    for result in results:
        writer.writerow(result.__dict__)


def write_table(results: list[BenchmarkResult]) -> None:
    headers = [
        "backend",
        "function",
        "calls",
        "mean_us",
        "median_us",
        "min_us",
        "max_us",
        "returned",
    ]
    rows = [
        [
            result.backend,
            result.function,
            str(result.calls),
            f"{result.mean_us:.3f}",
            f"{result.median_us:.3f}",
            f"{result.min_us:.3f}",
            f"{result.max_us:.3f}",
            str(result.returned),
        ]
        for result in results
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark arbitrage APIs for directly imported backend modules."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Benchmark CSV path. Defaults to {DEFAULT_DATA_PATH}.",
    )
    parser.add_argument(
        "--pair-column",
        help="Column containing pair strings such as C:USD-EUR or USD/EUR.",
    )
    parser.add_argument("--base-column", help="Column containing base/from symbols.")
    parser.add_argument("--quote-column", help="Column containing quote/to symbols.")
    parser.add_argument("--bid-column", help="Column containing bid prices.")
    parser.add_argument("--ask-column", help="Column containing ask prices.")
    parser.add_argument("--max-cycle-length", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument(
        "--backend",
        action="append",
        choices=sorted(BACKEND_MODULES),
        help="Backend name to benchmark. May be repeated. Defaults to available SIMD backends.",
    )
    parser.add_argument(
        "--include-generic",
        action="store_true",
        help="Also benchmark the generic backend when auto-discovering backends.",
    )
    parser.add_argument(
        "--force-unavailable",
        action="store_true",
        help="Run explicitly requested backends even when runtime detection says unavailable.",
    )
    parser.add_argument("--format", choices=("table", "csv"), default="table")
    args = parser.parse_args()
    if args.max_cycle_length < 3:
        parser.error("--max-cycle-length must be at least 3")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    books = load_books(
        args.data,
        pair_column=args.pair_column,
        base_column=args.base_column,
        quote_column=args.quote_column,
        bid_column=args.bid_column,
        ask_column=args.ask_column,
    )

    if args.backend:
        availability = backend_availability()
        backend_modules = []
        for backend in args.backend:
            compiled, available = availability.get(backend, (False, False))
            if not args.force_unavailable and not (compiled and available):
                print(
                    f"skipping {backend}: compiled={compiled} available={available}",
                    file=sys.stderr,
                )
                continue
            backend_modules.append((backend, BACKEND_MODULES[backend]))
    else:
        backend_modules = available_backend_modules(args.include_generic)

    if not backend_modules:
        raise SystemExit("no runnable backends selected")

    results: list[BenchmarkResult] = []
    for backend, module_name in backend_modules:
        results.extend(
            benchmark_backend(
                backend=backend,
                module_name=module_name,
                books=books,
                iterations=args.iterations,
                warmups=args.warmups,
                max_cycle_length=args.max_cycle_length,
                fee_bps=args.fee_bps,
            )
        )

    if args.format == "csv":
        write_csv(results)
    else:
        write_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
