#!/usr/bin/env python3
"""Benchmark the four streaming workloads highlighted in the documentation."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from negcycle import available_backends


BACKEND_MODULES = {
    "generic": "negcycle._generic",
    "sse": "negcycle._sse",
    "avx": "negcycle._avx",
    "avx2": "negcycle._avx2",
    "avx512": "negcycle._avx512",
}
BACKEND_RECORD_NAMES = {
    "generic": "generic",
    "sse": "x86_sse",
    "avx": "x86_avx",
    "avx2": "x86_avx2",
    "avx512": "x86_avx512",
}
RESULT_FIELDS = (
    "cold_build_us",
    "full_recompute_us",
    "two_leg_incremental_us",
    "single_leg_update_us",
)


@dataclass(frozen=True)
class QuoteRow:
    base: str
    quote: str
    bid: float
    ask: float


@dataclass(frozen=True)
class Iterations:
    cold: int
    full: int
    incremental: int
    single: int


@dataclass(frozen=True)
class Sample:
    cold_build_us: float
    full_recompute_us: float
    two_leg_incremental_us: float
    single_leg_update_us: float


def parse_pair(ticker: str) -> tuple[str, str]:
    value = ticker.split(":", 1)[-1]
    if "-" in value:
        base, quote = value.split("-", 1)
    elif "/" in value:
        base, quote = value.split("/", 1)
    else:
        raise ValueError(f"unsupported ticker format: {ticker!r}")
    if not base or not quote:
        raise ValueError(f"unsupported ticker format: {ticker!r}")
    return base, quote


def load_rows(path: Path) -> list[QuoteRow]:
    rows: list[QuoteRow] = []
    with path.expanduser().open(newline="") as handle:
        for record in csv.DictReader(handle):
            base, quote = parse_pair(record["ticker"])
            bid = float(record["bid_price"])
            ask = float(record["ask_price"])
            if bid > 0.0 and ask > 0.0 and bid <= ask:
                rows.append(QuoteRow(base, quote, bid, ask))
    if not rows:
        raise ValueError(f"no valid quote rows loaded from {path}")
    return rows


def update_factor(index: int, row_count: int) -> float:
    row_index = index % row_count
    pass_index = index // row_count
    return 1.0 + (
        1.0e-7 if (row_index + pass_index) % 2 == 0 else -1.0e-7
    )


def timed_us_per_call(iterations: int, callback: Callable[[int], object]) -> float:
    was_enabled = gc.isenabled()
    gc.disable()
    start = time.perf_counter_ns()
    try:
        for index in range(iterations):
            callback(index)
    finally:
        elapsed_ns = time.perf_counter_ns() - start
        if was_enabled:
            gc.enable()
    return elapsed_ns / iterations / 1_000.0


def populate_native(detector, rows: list[QuoteRow]) -> None:
    for row in rows:
        detector.add_book(row.base, row.quote, row.bid, row.ask, 0.0)


def benchmark_native_sample(
    detector_type,
    rows: list[QuoteRow],
    iterations: Iterations,
    max_cycle_length: int,
) -> Sample:
    def cold_build(_index: int):
        detector = detector_type()
        populate_native(detector, rows)
        return detector.find_best_arbitrage(max_cycle_length)

    cold_build_us = timed_us_per_call(iterations.cold, cold_build)

    full_detector = detector_type()
    populate_native(full_detector, rows)

    def full_recompute(index: int):
        row = rows[index % len(rows)]
        factor = update_factor(index, len(rows))
        full_detector.add_book(
            row.base,
            row.quote,
            row.bid * factor,
            row.ask * factor,
            0.0,
        )
        return full_detector.find_best_arbitrage(max_cycle_length)

    full_recompute_us = timed_us_per_call(iterations.full, full_recompute)

    incremental_detector = detector_type()
    populate_native(incremental_detector, rows)

    def two_leg_incremental(index: int):
        row = rows[index % len(rows)]
        factor = update_factor(index, len(rows))
        return incremental_detector.add_book_and_find_best_arbitrage(
            row.base,
            row.quote,
            row.bid * factor,
            row.ask * factor,
            0.0,
            max_cycle_length,
        )

    two_leg_incremental_us = timed_us_per_call(
        iterations.incremental,
        two_leg_incremental,
    )

    single_detector = detector_type()
    populate_native(single_detector, rows)

    def single_leg_update(index: int):
        row = rows[index % len(rows)]
        return single_detector.add_quote_and_find_best_arbitrage(
            row.base,
            row.quote,
            row.bid * update_factor(index, len(rows)),
            0.0,
            max_cycle_length,
        )

    single_leg_update_us = timed_us_per_call(iterations.single, single_leg_update)
    return Sample(
        cold_build_us=cold_build_us,
        full_recompute_us=full_recompute_us,
        two_leg_incremental_us=two_leg_incremental_us,
        single_leg_update_us=single_leg_update_us,
    )


def build_networkx_graph(nx, rows: list[QuoteRow]):
    graph = nx.DiGraph()
    for row in rows:
        graph.add_edge(row.base, row.quote, weight=-math.log(row.bid))
        graph.add_edge(row.quote, row.base, weight=-math.log(1.0 / row.ask))
    return graph


def find_best_networkx_cycle(nx, graph, max_cycle_length: int):
    best_weight = 0.0
    best_cycle = None
    for cycle in nx.simple_cycles(graph, length_bound=max_cycle_length):
        if len(cycle) < 2:
            continue
        weight = sum(
            graph[cycle[index]][cycle[(index + 1) % len(cycle)]]["weight"]
            for index in range(len(cycle))
        )
        if weight < best_weight:
            best_weight = weight
            best_cycle = cycle
    return best_cycle


def benchmark_networkx_sample(
    nx,
    rows: list[QuoteRow],
    iterations: Iterations,
    max_cycle_length: int,
) -> Sample:
    def cold_build(_index: int):
        graph = build_networkx_graph(nx, rows)
        return find_best_networkx_cycle(nx, graph, max_cycle_length)

    cold_build_us = timed_us_per_call(iterations.cold, cold_build)

    full_graph = build_networkx_graph(nx, rows)

    def full_recompute(index: int):
        row = rows[index % len(rows)]
        factor = update_factor(index, len(rows))
        full_graph.add_edge(
            row.base,
            row.quote,
            weight=-math.log(row.bid * factor),
        )
        full_graph.add_edge(
            row.quote,
            row.base,
            weight=-math.log(1.0 / (row.ask * factor)),
        )
        return find_best_networkx_cycle(nx, full_graph, max_cycle_length)

    full_recompute_us = timed_us_per_call(iterations.full, full_recompute)

    incremental_graph = build_networkx_graph(nx, rows)

    def two_leg_incremental(index: int):
        row = rows[index % len(rows)]
        factor = update_factor(index, len(rows))
        incremental_graph.add_edge(
            row.base,
            row.quote,
            weight=-math.log(row.bid * factor),
        )
        incremental_graph.add_edge(
            row.quote,
            row.base,
            weight=-math.log(1.0 / (row.ask * factor)),
        )
        return find_best_networkx_cycle(nx, incremental_graph, max_cycle_length)

    two_leg_incremental_us = timed_us_per_call(
        iterations.incremental,
        two_leg_incremental,
    )

    single_graph = build_networkx_graph(nx, rows)

    def single_leg_update(index: int):
        row = rows[index % len(rows)]
        single_graph.add_edge(
            row.base,
            row.quote,
            weight=-math.log(row.bid * update_factor(index, len(rows))),
        )
        return find_best_networkx_cycle(nx, single_graph, max_cycle_length)

    single_leg_update_us = timed_us_per_call(iterations.single, single_leg_update)
    return Sample(
        cold_build_us=cold_build_us,
        full_recompute_us=full_recompute_us,
        two_leg_incremental_us=two_leg_incremental_us,
        single_leg_update_us=single_leg_update_us,
    )


def selected_backends(requested: list[str] | None) -> list[str]:
    availability = {
        record.name: (record.compiled, record.available)
        for record in available_backends()
    }
    names = requested or list(BACKEND_MODULES)
    selected: list[str] = []
    for name in names:
        record_name = BACKEND_RECORD_NAMES[name]
        compiled, available = availability.get(record_name, (False, False))
        if compiled and available:
            selected.append(name)
        else:
            print(
                f"skipping {name}: compiled={compiled} available={available}",
                file=sys.stderr,
            )
    return selected


def summarize(samples: list[Sample]) -> Sample:
    return Sample(
        **{
            field: statistics.median(getattr(sample, field) for sample in samples)
            for field in RESULT_FIELDS
        }
    )


def log_samples(backend: str, samples: list[Sample]) -> None:
    for index, sample in enumerate(samples, start=1):
        values = " ".join(
            f"{field.removesuffix('_us')}={getattr(sample, field):.3f}"
            for field in RESULT_FIELDS
        )
        print(f"sample backend={backend} run={index} {values}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--backend",
        action="append",
        choices=sorted(BACKEND_MODULES),
        help="Native backend to run. May be repeated; defaults to all available x86 backends.",
    )
    parser.add_argument("--no-networkx", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-cycle-length", type=int, default=3)
    parser.add_argument("--native-cold-iterations", type=int, default=30)
    parser.add_argument("--native-full-iterations", type=int, default=2_000)
    parser.add_argument("--native-incremental-iterations", type=int, default=50_000)
    parser.add_argument("--native-single-iterations", type=int, default=100_000)
    parser.add_argument("--networkx-iterations", type=int, default=3)
    args = parser.parse_args()
    positive = (
        "repeats",
        "native_cold_iterations",
        "native_full_iterations",
        "native_incremental_iterations",
        "native_single_iterations",
        "networkx_iterations",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.max_cycle_length < 3:
        parser.error("--max-cycle-length must be at least 3")
    return args


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input)
    backends = selected_backends(args.backend)
    runners: list[tuple[str, Callable[[], Sample]]] = []

    if not args.no_networkx:
        try:
            import networkx as nx
        except ImportError as error:
            raise SystemExit(
                "NetworkX is required unless --no-networkx is passed"
            ) from error
        networkx_iterations = Iterations(
            cold=args.networkx_iterations,
            full=args.networkx_iterations,
            incremental=args.networkx_iterations,
            single=args.networkx_iterations,
        )
        runners.append(
            (
                "networkx",
                lambda: benchmark_networkx_sample(
                    nx,
                    rows,
                    networkx_iterations,
                    args.max_cycle_length,
                ),
            )
        )

    native_iterations = Iterations(
        cold=args.native_cold_iterations,
        full=args.native_full_iterations,
        incremental=args.native_incremental_iterations,
        single=args.native_single_iterations,
    )
    for backend in backends:
        detector_type = importlib.import_module(
            BACKEND_MODULES[backend]
        ).ArbitrageDetector
        runners.append(
            (
                backend,
                lambda detector_type=detector_type: benchmark_native_sample(
                    detector_type,
                    rows,
                    native_iterations,
                    args.max_cycle_length,
                ),
            )
        )

    writer = csv.writer(sys.stdout)
    writer.writerow(("backend", *RESULT_FIELDS))
    for backend, runner in runners:
        for _ in range(args.warmups):
            runner()
        samples = [runner() for _ in range(args.repeats)]
        log_samples(backend, samples)
        result = summarize(samples)
        writer.writerow(
            (
                backend,
                *(f"{getattr(result, field):.3f}" for field in RESULT_FIELDS),
            )
        )
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
