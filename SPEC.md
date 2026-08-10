# NegCycle Reconstruction Specification

## 1. Purpose and interpretation

This document is a self-contained build specification for recreating NegCycle.
An implementation that satisfies the required behavior, architecture, public
surface, platform matrix, tests, benchmarks, and documentation described here
should be substantively equivalent to the project captured by this
specification.

The words **MUST**, **SHOULD**, and **MAY** are normative:

- **MUST** identifies behavior required for compatibility.
- **SHOULD** identifies an architectural or quality requirement that may be
  changed only when the replacement is demonstrably equivalent.
- **MAY** identifies optional implementation freedom.

Correctness and public behavior take priority over matching incidental source
formatting. Benchmark values are reference measurements, not portable pass/fail
thresholds.

## 2. Product definition

NegCycle is a Python package with a C++ core that finds profitable simple
cycles in a directed graph of conversion rates. Its primary workload is a warm,
long-lived detector receiving repeated quote or bid/ask updates.

The package MUST provide:

1. Exact enumeration of all profitable simple cycles up to a caller-supplied
   maximum length.
2. Exact selection of the best such cycle with deterministic tie-breaking.
3. Full recomputation and incremental best-cycle APIs.
4. A generic scalar implementation and platform-specific SIMD modules behind
   one Python import.
5. Runtime CPU-feature dispatch, with AVX-512 preferred over AVX2 when both are
   compiled and supported.
6. Reproducible benchmark programs and checked-in result tables.
7. A dependency-free static documentation builder and Cloudflare Pages
   configuration.

NegCycle is not an exchange connector, execution engine, order-management
system, portfolio system, or complete trading application.

Project identity:

- Distribution and import name: `negcycle`
- Initial version: `0.1.0`
- Python requirement: 3.10 or newer
- Native language level: C++23
- License: Apache License 2.0
- Author metadata: Adam DePrince
- Creator attribution in the website:
  [Goblin Reactor](https://goblinreactor.com/)
- Repository URL:
  [github.com/adamdeprince/negcycle](https://github.com/adamdeprince/negcycle)

## 3. Required repository shape

The recreated repository SHOULD use this organization:

```text
negcycle/
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── BENCHMARKS.md
├── SPEC.md
├── LICENSE
├── poetry.lock
├── benchmarks.csv
├── headline_benchmarks.csv
├── package.json
├── wrangler.toml
├── benchmarks/
│   ├── benchmark_arbitrage_api.py
│   ├── headline_benchmark.py
│   └── quick_simd_benchmark.py
├── scripts/
│   └── build_docs.py
├── src/
│   ├── negcycle/
│   │   ├── __init__.py
│   │   └── py.typed
│   └── cpp/
│       ├── module.cpp
│       ├── detect/detect.cpp
│       ├── common/
│       │   ├── backend.h
│       │   ├── detector_base.h
│       │   ├── detector_base.cpp
│       │   ├── detector_bindings.h
│       │   ├── types.h
│       │   └── types_module.cpp
│       └── backends/
│           ├── generic.cpp
│           ├── generic.hpp
│           ├── x86_simd_search.hpp
│           ├── x86_sse.cpp
│           ├── x86_avx.cpp
│           ├── x86_avx2.cpp
│           ├── x86_avx512.cpp
│           ├── macos_arm64_neon.cpp
│           ├── linux_aarch64_asimd.cpp
│           ├── linux_aarch64_sve.cpp
│           ├── linux_loongarch64_lsx.cpp
│           ├── linux_loongarch64_lasx.cpp
│           ├── linux_powerpc64_vsx.cpp
│           └── linux_riscv64_rvv.cpp
├── tests/
│   ├── test_backend_dispatch.py
│   └── test_find_arbitrage.py
└── html/
    ├── index.html
    ├── styles.css
    ├── goblin.png
    ├── README.html
    ├── BENCHMARKS.html
    ├── SPEC.html
    ├── benchmarks.csv
    └── headline_benchmarks.csv
```

Generated build directories, wheels, virtual environments, and the external
market-data input MUST NOT be required source files.

## 4. Mathematical and data model

### 4.1 Directed rates

Each quote is a directed edge from symbol `u` to symbol `v` with:

```text
gross_rate = executable_rate
fee_fraction = fee_bps / 10_000
net_rate = gross_rate * (1 - fee_fraction)
weight = -log(net_rate)
```

Native calculations MUST use single-precision stored values (`float`). It is
acceptable to evaluate `log` in double precision before converting the result
to `float`, as the reference implementation does.

A bid/ask book `(base, quote, bid, ask, fee_bps)` MUST create or replace two
directed quotes:

```text
base  -> quote at bid
quote -> base  at 1 / ask
```

The same fee is independently applied to both directed legs.

### 4.2 Cycles and profitability

A cycle MUST:

- contain at least two directed edges;
- contain no repeated vertex except the closing copy of the starting vertex;
- have no more than `max_cycle_length` edges;
- follow only currently existing directed quotes.

For a closed path `v0, v1, ..., vk, v0`:

```text
total_weight = sum(edge.weight)
log_gain = -total_weight
gain_factor = product(edge.net_rate)
pct_return = (gain_factor - 1) * 100
```

A cycle is profitable if and only if `total_weight < 0.0`. The comparison is
strict; a zero-weight cycle is not arbitrage.

### 4.3 Canonical representation and ordering

Currency IDs MUST be assigned monotonically in first-seen order. Full cycle
enumeration MUST emit one rotational representation per directed cycle by
requiring the starting vertex to be the lowest ID in that cycle. Opposite
directions are distinct cycles.

Cycles MUST be sorted best-first using this order:

1. Lower `total_weight` wins when the difference exceeds `1e-7`.
2. Within that tolerance, the shorter cycle wins.
3. If still tied, the lexicographically smaller closed vector of integer
   vertex IDs wins.

The same ordering MUST select the result of `find_best_arbitrage`.

## 5. Public Python API

Importing the package MUST expose exactly these primary names:

```python
from negcycle import (
    ArbitrageDetector,
    Cycle,
    Edge,
    available_backends,
    backend_is_available,
    detect_best_backend,
)
```

`ArbitrageDetector` MUST be the detector class from the backend chosen once at
package-import time. All direct native backend modules MUST expose the same
three names: `ArbitrageDetector`, `Cycle`, and `Edge`.

### 5.1 `ArbitrageDetector`

The detector MUST support these methods and signatures:

```python
ArbitrageDetector()

detector.add_currency(code: str) -> int

detector.add_quote(
    from_code: str,
    to_code: str,
    executable_rate: float,
    fee_bps: float = 0.0,
) -> None

detector.add_book(
    base: str,
    quote: str,
    bid: float,
    ask: float,
    fee_bps: float = 0.0,
) -> None

detector.find_best_arbitrage(max_cycle_length: int) -> Cycle | None
detector.find_arbitrage(max_cycle_length: int) -> list[Cycle]

detector.add_quote_and_find_best_arbitrage(
    from_code: str,
    to_code: str,
    executable_rate: float,
    fee_bps: float,
    max_cycle_length: int,
) -> Cycle | None

detector.add_quote_and_find_arbitrage(
    from_code: str,
    to_code: str,
    executable_rate: float,
    fee_bps: float,
    max_cycle_length: int,
) -> list[Cycle]

detector.add_book_and_find_best_arbitrage(
    base: str,
    quote: str,
    bid: float,
    ask: float,
    fee_bps: float,
    max_cycle_length: int,
) -> Cycle | None

detector.add_book_and_find_arbitrage(
    base: str,
    quote: str,
    bid: float,
    ask: float,
    fee_bps: float,
    max_cycle_length: int,
) -> list[Cycle]
```

Every search method exposed to Python MUST reject `max_cycle_length < 3` with
`ValueError`. The native internals MAY support length two directly; a
`max_cycle_length` of three still includes profitable two-edge cycles.

The detector MUST also provide:

- read-only `currencies`, preserving insertion order;
- `len(detector)` equal to the number of currencies;
- `bool(detector)` false only when there are no currencies;
- iteration over currency strings in insertion order;
- membership by currency string;
- membership by `(from_symbol, to_symbol)` tuple;
- membership by `Edge`, based on its directed endpoints;
- `copy.copy`, `copy.deepcopy`, and pickle round-trips that retain currencies
  and quotes but need not retain computed caches;
- human-readable `str` and backend-specific `repr`.

Repeated insertion of a currency MUST return its existing ID. Repeated
insertion of a directed quote MUST replace that edge's rate and fee rather than
create a duplicate.

### 5.2 Massive currency helpers

Every usable detector class MUST provide:

```python
ArbitrageDetector.parse_massive_currency(pair: str) -> tuple[str, str]
```

Accepted strings MAY contain an arbitrary prefix ending in `:`, and the pair
itself MUST use either `BASE-QUOTE` or `BASE/QUOTE`. Surrounding ASCII
whitespace is ignored. Malformed pairs raise `ValueError`.

It MUST also provide:

```python
detector.process_massive_currency_bulk_file(
    file_obj,
    max_cycle_length: int = 5,
    fee_bps: float = 0.0,
) -> iterator
```

This helper reads text through `file_obj.readline()`. It parses at least the
first six comma-separated fields as ticker, an unused field, ask price, an
unused field, bid price, and participant timestamp in nanoseconds. A first
line whose ticker cannot be parsed is treated as a header. Blank lines are
ignored. Rows are stable-sorted by participant timestamp before replay.

For each replayed book update, the iterator MUST suppress `None` results and
yield profitable results as:

```text
(cycle, from_symbol, to_symbol, ask_price, bid_price, timestamp_seconds)
```

### 5.3 `Edge`

`Edge` MUST be a native shared type with read-only Python attributes:

- `from_symbol` and alias `from_`;
- `to_symbol` and alias `to`;
- `gross_rate`;
- `fee_bps`;
- `net_rate`;
- `weight`.

It MUST behave as a two-element sequence of endpoint strings:

- `len(edge) == 2`;
- `edge[0]` and `edge[-2]` are the source;
- `edge[1]` and `edge[-1]` are the destination;
- iteration is source then destination;
- `reversed(edge)` is destination then source;
- string membership tests either endpoint.

Exact field equality and hashing, string/repr formatting, and pickle support
MUST be implemented.

### 5.4 `Cycle`

`Cycle` MUST be a native shared type with read-only Python attributes:

- `vertices`: closed integer-ID path;
- `names`: closed symbol path parallel to `vertices`;
- `legs`: one `Edge` per directed leg;
- `length`: number of legs;
- `total_weight`;
- `log_gain`;
- `gain_factor`;
- `pct_return`.

It MUST behave as a sequence of legs, including negative indexing, forward and
reverse iteration, and `len(cycle) == cycle.length`. A `Cycle` object is always
truthy. Membership MUST support:

- symbol string: present anywhere in `names`;
- `(from_symbol, to_symbol)`: directed edge present in `legs`;
- `Edge`: its directed endpoints are present.

`cycle.same(other)` MUST return true when another `Cycle` has the same number
of legs and the same set of directed symbol pairs, independent of rotational
starting point. It returns false for `None` and non-cycle objects.

Exact structural equality and hashing, readable string/repr output, and pickle
support MUST be implemented.

### 5.5 Validation errors

The native layer MUST reject invalid input with exceptions translated to
Python `ValueError`:

- executable rate not greater than zero: `rate must be > 0`;
- fee outside `0 <= fee_bps < 10000`:
  `fee_bps must be in [0, 10000)`;
- bid or ask not positive, or bid greater than ask: `invalid bid/ask`;
- public search length below three:
  `max_cycle_length must be at least 3`.

## 6. Core storage and scalar search

The shared detector base SHOULD use the following model because it is central
to the update and SIMD performance characteristics:

- `vector<string>` for currency codes;
- `unordered_map<string, int>` for IDs;
- an `n * n` row-major matrix of quote cells;
- a sorted outgoing-adjacency vector per source vertex;
- absent matrix entries represented as non-existent with weight `+infinity`;
- cached dense weight and transposed-weight matrices;
- a cached best cycle associated with one `max_cycle_length`.

Adding a new currency expands the square quote matrix while preserving all
existing cells. Adding an existing edge updates the dense matrices in place
when their dimensions are still valid. Matrix resize invalidates dense caches.

The generic implementation MUST use depth-first enumeration of simple paths.
It MUST enforce canonical rotation by skipping candidate vertices whose ID is
less than or equal to the chosen start, except for explicit cycle closure.
It MUST materialize only negative-weight cycles.

## 7. Incremental best-cycle algorithm

Incremental APIs MUST be exact; they are not approximate shortcuts.

### 7.1 Single directed quote

After an edge upsert:

1. Perform a full best-cycle search if there is no valid cache, the requested
   maximum length differs from the cached length, or a new currency was added.
2. Classify a weight decrease greater than `1e-7`, or a new edge, as an
   improvement.
3. Classify a weight increase greater than `1e-7` as a worsening.
4. For an improved edge, search only cycles containing that edge. Preserve and
   compare the cached cycle when it did not use the changed edge.
5. For a worsened edge, return the cached cycle unchanged if it did not use
   the edge; otherwise run a full search.
6. For an effectively unchanged edge, return the cache.

The fixed-edge search starts with the changed directed edge and enumerates a
simple return path to its source.

### 7.2 Two-leg book update

The book update MUST compare the forward bid edge and reverse `1 / ask` edge
against stored gross rates and fees with tolerance `1e-7`.

- If both legs are unchanged, return a compatible cache or compute it once.
- If only one leg changed, route through the single-edge incremental path.
- If a worsened leg belongs to the cached best cycle, perform a full search.
- Otherwise, search cycles through every improved leg and compare them with an
  unaffected cached cycle.

### 7.3 All-cycle update methods

The methods returning `list[Cycle]` MUST upsert the quote or book and perform a
complete exact enumeration. Only the best-cycle update methods use the cache
optimization above.

## 8. SIMD search architecture

### 8.1 Common strategy

SIMD detector classes MUST reuse the shared state and public bindings. Their
specialized search SHOULD enumerate path prefixes scalarly and vectorize the
last intermediate vertex by adding:

```text
prefix_weight + weight[current, candidate] + weight[candidate, start]
```

across a dense row and the corresponding transposed row.

The optimized fixed-width/SVE search MUST handle cycle lengths up to five. It
MUST fall back to the scalar exact implementation for larger maximum lengths.
All SIMD results MUST have the same cycle ordering and values, within normal
floating-point tolerance, as the generic implementation.

Dense float buffers SHOULD be 64-byte aligned. Store both the row-major weight
matrix and its transpose so closing-edge loads are contiguous.

### 8.2 Loose scan policy

SSE, AVX, AVX2, NEON/ASIMD, LSX, and LASX use an unpadded dense stride. Their
last-hop scan processes complete vectors and then a scalar tail. The hit
threshold is zero, so it identifies every profitable candidate before the
common tie-breaker runs.

### 8.3 AVX-512 policy

AVX-512 MUST use 16 `float` lanes and a padded/tight scan policy:

- round the dense row stride up far enough to load a final 16-lane block that
  covers every real candidate;
- fill padding with `+infinity`;
- eliminate the scalar tail;
- if a best candidate exists, use
  `min(best_weight + 1e-7, 0)` as the pruning threshold;
- fold that threshold into the broadcast prefix while retaining candidates
  close enough for length and lexicographic tie-breaking.

On an AVX-512-capable x86 machine, this backend MUST be the default.

### 8.4 SVE policy

SVE MUST use the same padded/tight semantics with runtime SVE predicates for a
partial vector tail. A 16-float padding multiple preserves 64-byte row
alignment, but the implementation MUST not assume a fixed SVE vector length.

## 9. Backend catalog, modules, and dispatch

The native backend enum and records MUST contain these names:

| Backend kind | Module | Build condition | Runtime feature | State |
| --- | --- | --- | --- | --- |
| `generic` | `negcycle._generic` | all builds | none | functional scalar |
| `x86_sse` | `negcycle._sse` | Linux x86-64 | `X86_SSE2` | functional SIMD |
| `x86_avx` | `negcycle._avx` | Linux x86-64 | `X86_AVX` | functional SIMD |
| `x86_avx2` | `negcycle._avx2` | Linux x86-64 | `X86_AVX2` | functional SIMD |
| `x86_avx512` | `negcycle._avx512` | Linux x86-64 | `X86_AVX512F` | functional SIMD |
| `macos_arm64_neon` | `negcycle._macos_arm64_neon` | macOS ARM64 | `ARM_ASIMD` | functional SIMD |
| `linux_aarch64_asimd` | `negcycle._linux_aarch64_asimd` | Linux AArch64 | `ARM_ASIMD` | functional SIMD |
| `linux_aarch64_sve` | `negcycle._linux_aarch64_sve` | Linux AArch64 | `ARM_SVE` | functional SIMD |
| `linux_loongarch64_lsx` | `negcycle._linux_loongarch64_lsx` | Linux LoongArch64 | `LA_LSX` | functional SIMD |
| `linux_loongarch64_lasx` | `negcycle._linux_loongarch64_lasx` | Linux LoongArch64 | `LA_LASX` | functional SIMD |
| `linux_powerpc64_vsx` | `negcycle._linux_powerpc64_vsx` (reserved, not built) | Linux POWER64 | `PPC_VSX` | catalog/scaffold only |
| `linux_riscv64_rvv` | `negcycle._linux_riscv64_rvv` (reserved, not built) | Linux RISC-V 64 | `RV_V` | catalog/scaffold only |

POWER VSX and RISC-V RVV source files currently reserve the backend names and
contain implementation TODOs; they are not required to expose detector
extension modules. Recreating them as placeholders is substantively correct.

Runtime feature detection MUST use `goblin-cpu-features>=0.1.0`. If that
dependency cannot be imported, only generic can be considered available.

The exact default priority MUST be:

```text
x86_avx512
x86_avx2
x86_avx
x86_sse
macos_arm64_neon
linux_aarch64_sve
linux_aarch64_asimd
linux_loongarch64_lasx
linux_loongarch64_lsx
linux_powerpc64_vsx
linux_riscv64_rvv
generic
```

`available_backends()` MUST return compiled backend records with `available`
filled by the Python CPU-feature check. `backend_is_available(kind)` MUST
require both a known module mapping and its feature set, except that generic is
always available. `detect_best_backend()` MUST return the first available kind
in the priority list.

Loading the selected module MUST catch `ImportError` and fall back to the
generic detector class. Direct imports of functional backend modules MUST
remain possible for benchmarking and parity tests.

## 10. Native module boundaries and GIL behavior

The build MUST create separate nanobind extension modules:

- `_common` for the one shared definition of `Edge` and `Cycle`;
- `_negcycle_native` for `BackendKind`, `BackendRecord`, and compiled-backend
  discovery;
- `_generic` and each functional platform backend for its detector class.

Every detector module MUST import and re-export the types from `_common`; it
MUST NOT bind incompatible per-module copies of `Edge` or `Cycle`.

Full `find_best_arbitrage` and `find_arbitrage` calls MUST release the Python
GIL. Incremental calls with `max_cycle_length > 3` MUST release it. The shortest
incremental calls SHOULD retain it to avoid release/reacquire overhead. A
single detector instance is mutable and need not promise safe concurrent use.

## 11. Build and packaging

### 11.1 Python build metadata

Use PEP 517 with:

- `scikit-build-core>=0.10` as build backend;
- `nanobind>=2.0`;
- `cmake>=3.26`;
- `ninja>=1.11`.

Runtime dependency:

```text
goblin-cpu-features>=0.1.0
```

Development extras MUST include pytest, mypy, and Ruff. The benchmark extra
MUST include `massive-speedup>=0.1.4`. NetworkX is required only for the
headline comparison and MAY remain an explicitly installed benchmark tool.

The project SHOULD build an ABI3 wheel using a CPython 3.10 limited-API floor.
Editable installs SHOULD use scikit-build-core redirect mode.

### 11.2 CMake rules

CMake MUST:

- require version 3.26 or newer;
- require C++23 with extensions disabled;
- build all configurations at `-O3`;
- discover Python's interpreter and extension-module development component;
- use nanobind modules without size optimization;
- install native libraries into the `negcycle` package.

Platform-specific compile flags:

| Module | Required flags |
| --- | --- |
| x86 SSE | `-msse2` |
| x86 AVX | `-mavx -mtune=core-avx-i` |
| x86 AVX2 | `-mavx2` |
| x86 AVX-512 | `-mavx512f` |
| Linux AArch64 generic/common | `-march=armv8-a` |
| Linux ASIMD | `-march=armv8-a+simd` |
| Linux SVE | `-march=armv8.2-a+sve` |
| LoongArch generic/common | `-march=loongarch64 -mno-lsx -mno-lasx` |
| LoongArch LSX | `-march=loongarch64 -mlsx -mno-lasx` |
| LoongArch LASX | `-march=loongarch64 -mlsx -mlasx` |

LoongArch targets SHOULD statically link libstdc++ and libgcc. Baseline import
modules MUST never contain instructions from a higher optional ISA.

Recommended developer commands:

```bash
poetry install --with dev
poetry run pip install --no-build-isolation -ve .
poetry run pytest
```

A non-isolated wheel helper MAY run:

```bash
poetry run python -m build --wheel --no-isolation
```

## 12. Required correctness tests

At minimum, pytest MUST cover the following acceptance cases.

### 12.1 Exact result and ordering

Insert these directed rates in this order:

```python
USD -> EUR: 1.2
EUR -> USD: 1.1
USD -> JPY: 1.1
JPY -> USD: 1.1
EUR -> JPY: 1.1
JPY -> EUR: 1.1
```

`find_arbitrage(3)` MUST return cycle names in this order:

```python
[
    ("USD", "EUR", "JPY", "USD"),
    ("USD", "JPY", "EUR", "USD"),
    ("USD", "EUR", "USD"),
    ("USD", "JPY", "USD"),
    ("EUR", "JPY", "EUR"),
]
```

Every result must have negative total weight, the weights must be sorted
ascending, and `find_best_arbitrage(3)` must match the first result.

### 12.2 Update behavior

Starting empty, the first two legs of a profitable triangle MUST return no
cycles from `add_quote_and_find_arbitrage`; the closing third leg MUST return
the triangle. Adding an unrelated non-profitable book MUST leave that result
unchanged.

Tests SHOULD additionally compare incremental best-cycle results against a
fresh full recomputation after improvements, worsenings, unchanged updates,
new edges, new currencies, and changed maximum lengths.

### 12.3 Backend parity

For every compiled and runtime-available functional backend, directly import
its detector class and compare both all-cycle and best-cycle results with the
generic backend on the same graph.

### 12.4 Dispatch

Mock compiled backends and CPU features to prove:

- AVX-512 is selected when AVX2 and AVX-512F are both available;
- AVX2 is selected when AVX512F is absent;
- generic is the final safe fallback.

## 13. Benchmark programs and reference data

Benchmark code is part of the product. Timing assertions MUST NOT be part of
unit tests.

### 13.1 Compact input format

The compact CSV readers MUST accept named columns:

```text
ticker,bid_price,ask_price,...
```

Ticker parsing accepts an optional prefix before `:` and either `BASE-QUOTE`
or `BASE/QUOTE`. Skip rows where bid or ask is non-positive or bid exceeds ask.

The reference `benchmark_data.csv` is external rather than committed. It has:

- 1,206 data rows and 1,207 lines including its header;
- 1,206 unique tickers and 129 currencies;
- SHA-256
  `ef8b149ba0a312bddcdf0ec1b77be47cd8abbe4898292fa540945e156aa4b6bf`.

### 13.2 Headline benchmark

`benchmarks/headline_benchmark.py` MUST directly benchmark generic and all
available x86 modules plus an exact NetworkX baseline. It MUST implement four
modes:

1. cold detector construction, population, and one best-cycle computation;
2. a warm book update followed by full recomputation;
3. a warm two-leg incremental update;
4. a warm single-leg incremental update.

The update factor alternates by `1 ± 1e-7`. Disable garbage collection during
timed loops and use `perf_counter_ns`. Defaults:

- seven recorded repeats;
- one discarded warm-up;
- maximum cycle length three;
- native iterations: 30 cold, 2,000 full, 50,000 two-leg, 100,000 single-leg;
- NetworkX iterations: three for each mode.

NetworkX MUST use a `DiGraph` and enumerate exact bounded simple cycles with
`simple_cycles(length_bound=3)` on every measured pass.

Reference command on Linux:

```bash
taskset -c 0 python benchmarks/headline_benchmark.py benchmark_data.csv
```

Reference medians from an Intel Xeon 6975P-C 4-vCPU KVM guest are in
microseconds per pass:

| Backend | single-leg incremental | two-leg incremental | full recompute | cold build + compute |
| --- | ---: | ---: | ---: | ---: |
| networkx | 87296.218 | 87252.850 | 87251.911 | 88429.978 |
| generic | 0.643 | 1.185 | 89.019 | 2736.103 |
| SSE | 0.460 | 0.709 | 36.746 | 2677.078 |
| AVX | 0.451 | 0.653 | 20.547 | 2659.697 |
| AVX2 | 0.441 | 0.647 | 18.720 | 2654.793 |
| AVX-512 | 0.445 | 0.641 | 12.861 | 2630.359 |

The documentation MUST treat the roughly 1% AVX2/AVX-512 differences on the
two incremental paths as noise. It MUST call out the material full-recompute
result: AVX-512 reduces 18.720 microseconds to 12.861 microseconds, about 31%.

### 13.3 Quick SIMD throughput benchmark

`benchmarks/quick_simd_benchmark.py` MUST:

- load the compact CSV into memory using the standard library;
- discover or accept explicit backend modules;
- benchmark the four quote/book and best/all update combinations;
- alternate updates by `1 ± 1e-7`;
- use ten times the requested base pass count for best-cycle variants and half
  the pass count for all-cycle variants;
- report calls, elapsed seconds, calls/sec, mean microseconds, result count,
  and speedup against generic.

Reference command:

```bash
taskset -c 0 python benchmarks/quick_simd_benchmark.py benchmark_data.csv --passes 20
```

Run it seven times after one warm-up and aggregate median calls/sec. The Intel
reference generic rates are 2,812,169 quote-best, 6,463 quote-all, 841,367
book-best, and 4,927 book-all calls/sec. AVX-512 reaches 1.75x generic for
quote-all and 1.70x for book-all. These calls/sec measurements are separate
from the headline microseconds/pass comparison.

### 13.4 Full replay benchmark

`benchmarks/benchmark_arbitrage_api.py` MUST use the optional
`massive-speedup` package to replay Massive currency quote flat files. It MUST:

- time a parser-only pass to measure input overhead and graph shape;
- benchmark any selected available backend and update function;
- support row limits, optional participant-timestamp sorting, fee and maximum
  length controls;
- output either a readable table or CSV;
- report raw and parser-overhead-subtracted latency and throughput.

### 13.5 Checked-in results

Keep `headline_benchmarks.csv` for the six headline median rows and their
provenance. Keep `benchmarks.csv` for backend/function throughput rows across
Intel, Apple ARM64, and Loongson. Preserve disclosed limitations, including
LASX measuring 0.88x generic for the small quote-best workload.

## 14. Documentation website

### 14.1 Static site

The public site MUST be static and live in `html/`. It MUST require no browser
framework and no documentation generator dependency.

The hand-authored `html/index.html` MUST:

- lead with the performance case for caring, not an API reference;
- use a hero with the three AVX-512 headline results and NetworkX comparisons;
- identify benchmark conditions without exposing machine host aliases;
- put the benchmark table immediately after the hero, with incremental columns
  first and cold build last;
- explain that AVX-512's 31% full-recompute gain is material and the roughly 1%
  incremental gaps are noise;
- describe runtime dispatch and AVX-512-first x86 priority;
- list all architecture backend kinds while clearly allowing POWER/RVV to
  remain future scaffolds in technical documentation;
- preserve an honest limitations panel, including the Loongson LASX result;
- include benchmark provenance, reproduction commands, and links to both CSVs;
- place Quick Start, Incremental API, Book Updates, Build, and Repository Layout
  below the benchmark/backend material;
- state that the project is not a full execution stack, venue adapter, or order
  management system;
- link every generated top-level Markdown page;
- display `goblin.png` as the mascot and credit Goblin Reactor.

All examples SHOULD use the actual API names `cycle.legs`, `edge.from_symbol`,
and `edge.to_symbol`.

### 14.2 Visual language

`html/styles.css` MUST provide the same broad visual identity:

- warm cream/paper background with rust/orange accents and dark brown text;
- serif editorial body type and sans-serif labels;
- rounded translucent cards with subtle borders and shadows;
- two-column responsive panel grids;
- a dark terminal-style code panel;
- responsive benchmark tables with horizontal overflow;
- a three-item hero stat strip collapsing to one column on narrow screens;
- responsive mascot/document panels;
- modest rise-in animation only when reduced motion is not requested.

Reuse semantic classes such as `page-shell`, `hero`, `hero-copy`, `hero-card`,
`panel`, `panel-wide`, `grid-section`, `section-label`, `feature-list`,
`code-block`, `table-wrap`, `stat-strip`, `stat-item`, `mascot-panel`, and
`document-content`.

### 14.3 Markdown compiler

`scripts/build_docs.py` MUST use only the Python standard library. On every
build it MUST:

1. Find all top-level `*.md` files, with README first and the rest
   case-insensitively sorted.
2. Render each to `html/<stem>.html` using the common stylesheet and panel
   shell.
3. Support headings, fenced code, paragraphs, ordered/unordered lists,
   blockquotes, pipe tables, links, inline code, bold, and italic markup.
4. HTML-escape source content.
5. Replace only the content between explicit generated-document-link markers
   in `html/index.html`.
6. Copy `benchmarks.csv` and `headline_benchmarks.csv` into `html/`.

Generated pages MUST link back to the landing page and credit Goblin Reactor.

### 14.4 Cloudflare Pages

Use this dependency-free package script contract:

```json
{
  "name": "negcycle-docs",
  "private": true,
  "scripts": {
    "build": "python3 scripts/build_docs.py",
    "preview": "npm run build && npx wrangler pages dev",
    "deploy": "npm run build && npx wrangler pages deploy --branch=main"
  }
}
```

`wrangler.toml` MUST identify project `negcycle`, set
`pages_build_output_dir = "./html"`, and use an explicit compatibility date.

Cloudflare configuration:

```text
Build command: npm run build
Build output directory: html
Deploy command: npm run deploy
```

## 15. Completion checklist

An AI implementation is complete only when all of the following are true:

- The package builds from a clean checkout with the declared toolchain.
- `from negcycle import ArbitrageDetector` succeeds on supported platforms.
- The mathematical model, fee handling, book conversion, cycle
  canonicalization, sorting, and validation rules match this specification.
- Incremental best-cycle methods match fresh full recomputation in correctness
  tests.
- Every available native backend matches generic results.
- Mocked dispatch selects AVX-512 ahead of AVX2 and falls back safely.
- Shared native types work across every detector module.
- Copying and pickling the detector, edges, and cycles work.
- The three benchmark programs run and report the documented units without
  conflating microseconds/pass and calls/sec.
- `npm run build` regenerates every top-level Markdown page, updates document
  links, and copies result CSVs.
- Every local link, image, stylesheet, and fragment in generated HTML resolves.
- The landing page contains no benchmark-machine host aliases.
- The Apache-2.0 license and Goblin Reactor attribution are present.

Performance should be evaluated on representative hardware, but correctness
MUST never depend on attaining the historical timing values above.
