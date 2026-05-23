#!/usr/bin/env python3
"""Replay Massive currency quotes through arbitrage update APIs."""

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

from negcycle import available_backends


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

SIMD_BACKENDS = {
    "x86_sse",
    "x86_avx",
    "x86_avx2",
    "x86_avx512",
    "macos_arm64_neon",
    "linux_aarch64_asimd",
    "linux_aarch64_sve",
    "linux_loongarch64_lsx",
    "linux_loongarch64_lasx",
}

UPDATE_FUNCTIONS = (
    "add_quote_and_find_best_arbitrage",
    "add_quote_and_find_arbitrage",
    "add_book_and_find_best_arbitrage",
    "add_book_and_find_arbitrage",
)


@dataclass(frozen=True)
class QuoteUpdate:
    base: str
    quote: str
    bid: float
    ask: float


@dataclass
class PassStats:
    calls: int = 0
    skipped: int = 0
    currencies: set[str] | None = None
    pairs: set[tuple[str, str]] | None = None
    returned: int = 0


@dataclass(frozen=True)
class BenchmarkResult:
    date: str
    input_path: str
    backend: str
    module: str
    function: str
    calls: int
    skipped: int
    currencies: int
    pairs: int
    max_cycle_length: int
    total_seconds: float
    mean_us: float
    calls_per_second: float
    returned: int
    overhead_seconds: float
    net_seconds: float
    net_mean_us: float
    net_calls_per_second: float


def parse_pair(ticker: str) -> tuple[str, str]:
    if ":" in ticker:
        ticker = ticker.split(":", 1)[1]

    if "-" in ticker:
        base, quote = ticker.split("-", 1)
    elif "/" in ticker:
        base, quote = ticker.split("/", 1)
    else:
        raise ValueError(f"unsupported currency ticker format: {ticker!r}")

    if not base or not quote:
        raise ValueError(f"unsupported currency ticker format: {ticker!r}")
    return base, quote


def resolve_input_path(input_path: Path) -> Path:
    path = input_path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def infer_date_from_path(path: Path) -> str:
    for suffix in (".csv.gz", ".csv"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def iter_quote_updates(
    path: Path,
    *,
    limit: int | None,
    sort_by_participant_timestamp: bool,
):
    try:
        import massive_speedup
    except ImportError as error:
        raise RuntimeError(
            "benchmark_arbitrage_api.py requires massive-speedup; install with "
            "`python -m pip install massive-speedup` or the `bench` extra."
        ) from error

    rows = massive_speedup.FlatFiles.currency.Quote.parse(
        path,
        sort_by_participant_timestamp=sort_by_participant_timestamp,
    )
    yielded = 0
    for row in rows:
        base, quote = parse_pair(row.ticker)
        bid = float(row.bid_price)
        ask = float(row.ask_price)
        update = QuoteUpdate(base=base, quote=quote, bid=bid, ask=ask)
        yielded += 1
        yield update
        if limit is not None and yielded >= limit:
            break


def is_valid_update(update: QuoteUpdate) -> bool:
    return update.bid > 0.0 and update.ask > 0.0 and update.bid <= update.ask


def count_returned(value) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    return 1


def time_stream_pass(
    *,
    path: Path,
    limit: int | None,
    sort_by_participant_timestamp: bool,
    handler: Callable[[QuoteUpdate], object] | None,
    collect_shape: bool,
) -> tuple[float, PassStats]:
    stats = PassStats(
        currencies=set() if collect_shape else None,
        pairs=set() if collect_shape else None,
    )
    was_enabled = gc.isenabled()
    gc.disable()
    start = time.perf_counter()
    try:
        for update in iter_quote_updates(
            path,
            limit=limit,
            sort_by_participant_timestamp=sort_by_participant_timestamp,
        ):
            if collect_shape:
                assert stats.currencies is not None
                assert stats.pairs is not None
                stats.currencies.add(update.base)
                stats.currencies.add(update.quote)
                stats.pairs.add((update.base, update.quote))

            if not is_valid_update(update):
                stats.skipped += 1
                continue

            if handler is None:
                value = None
            else:
                value = handler(update)
            stats.calls += 1
            stats.returned = count_returned(value)
    finally:
        total_seconds = time.perf_counter() - start
        if was_enabled:
            gc.enable()
    return total_seconds, stats


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
        if module_name is not None:
            modules.append((record.name, module_name))

    modules.sort(key=lambda item: (item[0] != "generic", item[0]))
    return modules


def load_backend_detector(module_name: str):
    module = importlib.import_module(module_name)
    return module.ArbitrageDetector


def make_handler(
    detector,
    function: str,
    *,
    fee_bps: float,
    max_cycle_length: int,
) -> Callable[[QuoteUpdate], object]:
    method = getattr(detector, function)
    if function.startswith("add_quote"):
        return lambda update: method(
            update.base,
            update.quote,
            update.bid,
            fee_bps,
            max_cycle_length,
        )
    return lambda update: method(
        update.base,
        update.quote,
        update.bid,
        update.ask,
        fee_bps,
        max_cycle_length,
    )


def summarize(
    *,
    date: str,
    input_path: Path,
    backend: str,
    module_name: str,
    function: str,
    max_cycle_length: int,
    total_seconds: float,
    stats: PassStats,
    currencies: int,
    pairs: int,
    overhead_seconds: float,
) -> BenchmarkResult:
    mean_us = (total_seconds / stats.calls * 1.0e6) if stats.calls else 0.0
    calls_per_second = stats.calls / total_seconds if total_seconds > 0 else 0.0
    net_seconds = total_seconds - overhead_seconds
    net_mean_us = (net_seconds / stats.calls * 1.0e6) if stats.calls else 0.0
    net_calls_per_second = stats.calls / net_seconds if net_seconds > 0 else 0.0
    return BenchmarkResult(
        date=date,
        input_path=str(input_path),
        backend=backend,
        module=module_name,
        function=function,
        calls=stats.calls,
        skipped=stats.skipped,
        currencies=currencies,
        pairs=pairs,
        max_cycle_length=max_cycle_length,
        total_seconds=total_seconds,
        mean_us=mean_us,
        calls_per_second=calls_per_second,
        returned=stats.returned,
        overhead_seconds=overhead_seconds,
        net_seconds=net_seconds,
        net_mean_us=net_mean_us,
        net_calls_per_second=net_calls_per_second,
    )


def benchmark_backend(
    *,
    date: str,
    input_path: Path,
    backend: str,
    module_name: str,
    functions: tuple[str, ...],
    fee_bps: float,
    max_cycle_length: int,
    limit: int | None,
    sort_by_participant_timestamp: bool,
    overhead_seconds: float,
    currencies: int,
    pairs: int,
    emit: Callable[[BenchmarkResult], None],
) -> None:
    detector_type = load_backend_detector(module_name)
    for function in functions:
        detector = detector_type()
        handler = make_handler(
            detector,
            function,
            fee_bps=fee_bps,
            max_cycle_length=max_cycle_length,
        )
        total_seconds, stats = time_stream_pass(
            path=input_path,
            limit=limit,
            sort_by_participant_timestamp=sort_by_participant_timestamp,
            handler=handler,
            collect_shape=False,
        )
        emit(
            summarize(
                date=date,
                input_path=input_path,
                backend=backend,
                module_name=module_name,
                function=function,
                max_cycle_length=max_cycle_length,
                total_seconds=total_seconds,
                stats=stats,
                currencies=currencies,
                pairs=pairs,
                overhead_seconds=overhead_seconds,
            )
        )


TABLE_HEADERS = [
    "backend",
    "function",
    "calls",
    "seconds",
    "mean_us",
    "net_us",
    "calls_s",
    "net_calls_s",
    "returned",
]
TABLE_WIDTHS = [16, 35, 10, 12, 12, 12, 10, 12, 10]


def table_values(result: BenchmarkResult) -> list[str]:
    return [
        result.backend,
        result.function,
        str(result.calls),
        f"{result.total_seconds:.6f}",
        f"{result.mean_us:.3f}",
        f"{result.net_mean_us:.3f}",
        f"{result.calls_per_second:.0f}",
        f"{result.net_calls_per_second:.0f}",
        str(result.returned),
    ]


def format_table_line(values: list[str]) -> str:
    padded: list[str] = []
    for index, value in enumerate(values):
        width = TABLE_WIDTHS[index]
        if index < 2:
            padded.append(value[:width].ljust(width))
        else:
            padded.append(value[:width].rjust(width))
    return "  ".join(padded)


class ResultWriter:
    def __init__(self, output_format: str) -> None:
        self.output_format = output_format
        self.csv_writer: csv.DictWriter | None = None
        if output_format == "csv":
            self.csv_writer = csv.DictWriter(
                sys.stdout,
                fieldnames=list(BenchmarkResult.__annotations__),
            )
            self.csv_writer.writeheader()
            sys.stdout.flush()
        else:
            print(format_table_line(TABLE_HEADERS), flush=True)
            print(format_table_line(["-" * width for width in TABLE_WIDTHS]), flush=True)

    def write(self, result: BenchmarkResult) -> None:
        if self.csv_writer is not None:
            self.csv_writer.writerow(result.__dict__)
            sys.stdout.flush()
        else:
            print(format_table_line(table_values(result)), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark arbitrage update APIs by replaying Massive currency quote "
            "flat-file CSV gzip data."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Massive currency quotes CSV gzip path.",
    )
    parser.add_argument("--max-cycle-length", type=int, default=3)
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
    parser.add_argument(
        "--function",
        action="append",
        choices=UPDATE_FUNCTIONS,
        help="Update function to benchmark. May be repeated. Defaults to all update functions.",
    )
    parser.add_argument(
        "--sort-by-participant-timestamp",
        action="store_true",
        help="Ask massive-speedup to sort the CSV by participant timestamp before replay.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit parsed quote rows. Intended for smoke tests.",
    )
    parser.add_argument("--format", choices=("table", "csv"), default="table")
    args = parser.parse_args()
    if args.max_cycle_length < 3:
        parser.error("--max-cycle-length must be at least 3")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main() -> int:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    date = infer_date_from_path(input_path)
    functions = tuple(args.function) if args.function else UPDATE_FUNCTIONS

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

    overhead_seconds, overhead_stats = time_stream_pass(
        path=input_path,
        limit=args.limit,
        sort_by_participant_timestamp=args.sort_by_participant_timestamp,
        handler=None,
        collect_shape=True,
    )
    currencies = len(overhead_stats.currencies or ())
    pairs = len(overhead_stats.pairs or ())

    overhead_result = summarize(
        date=date,
        input_path=input_path,
        backend="massive_speedup",
        module_name="massive_speedup",
        function="parse_currency_quotes",
        max_cycle_length=args.max_cycle_length,
        total_seconds=overhead_seconds,
        stats=overhead_stats,
        currencies=currencies,
        pairs=pairs,
        overhead_seconds=overhead_seconds,
    )

    writer = ResultWriter(args.format)
    writer.write(overhead_result)
    for backend, module_name in backend_modules:
        benchmark_backend(
            date=date,
            input_path=input_path,
            backend=backend,
            module_name=module_name,
            functions=functions,
            fee_bps=args.fee_bps,
            max_cycle_length=args.max_cycle_length,
            limit=args.limit,
            sort_by_participant_timestamp=args.sort_by_participant_timestamp,
            overhead_seconds=overhead_seconds,
            currencies=currencies,
            pairs=pairs,
            emit=writer.write,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
