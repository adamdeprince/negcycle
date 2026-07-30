# NegCycle

NegCycle is a Python library, with a high-performance C++ core, for finding the most profitable arbitrage cycle in a directed graph of exchange rates.

Give it a set of currency pair prices, or more generally any directed conversion rates, and NegCycle searches for the best cycle whose product of rates is greater than one, after any fees you choose to model. In graph theory terms, it finds the most negative cycle in the `-log(rate)` transformed graph, subject to a maximum cycle length.

The project is built for a practical use case: fast repeated searches over a changing market graph. It supports both full recomputation and incremental update workflows, so you can either build a detector from scratch and search once, or update one quote or one bid/ask book and ask for the new best opportunity.

NegCycle is designed to be:

- **fast**: the core search is implemented in C++, with scalar and SIMD backends
- **practical**: it exposes a simple Python API for loading quotes and books
- **exact**: it searches for the best simple cycle, avoiding repeated-node nonsense
- **benchmarkable**: it is easy to compare against pure Python and graph-library baselines

This project is still small and focused. The goal is not to be a full trading system. The goal is to provide a clean, fast, open-source engine for arbitrage cycle detection.

# NegCycle

NegCycle is a Python library, with a high-performance C++ core, for finding the most profitable arbitrage cycle in a directed graph of exchange rates.

Give it a set of currency pair prices, or more generally any directed conversion rates, and NegCycle searches for the best cycle whose product of rates is greater than one, after any fees you choose to model. In graph terms, it finds the most negative cycle in the `-log(rate)` transformed graph, subject to a maximum cycle length.

The project is built for a practical use case: fast repeated searches over a changing market graph. It supports both full recomputation and incremental update workflows, so you can either build a detector from scratch and search once, or update one quote or one bid/ask book and ask for the new best opportunity.

NegCycle is designed to be:

- **fast**: the core search is implemented in C++, with scalar and SIMD backends
- **practical**: it exposes a simple Python API for loading quotes and books
- **exact**: it searches for the best simple cycle, avoiding repeated-node nonsense
- **benchmarkable**: it is easy to compare against pure Python and graph-library baselines

This project is still small and focused. The goal is not to be a full trading system. The goal is to provide a clean, fast, open-source engine for arbitrage cycle detection.


## Performance

NegCycle is built for **streaming, low-latency** use.

The main design target is not one giant batch job. It is the common market-data workflow where prices keep changing and you want to repeatedly ask:

- what is the best arbitrage cycle right now?
- did one changed book create a better cycle?
- can I update an existing detector instead of rebuilding everything?

The current benchmark uses **real market data** across **1,206 bid/ask
currency-pair rows** and looks only at **3-currency cycles**. All times below
are in **microseconds per pass**. The benchmark was rerun on 2026-07-29 on a
4-vCPU Intel Xeon 6975P-C with AVX-512, using Python 3.14.4, GCC 16.1.0, and
NegCycle commit `37226cb`.

The four benchmark modes were:

1. **Cold build + single compute**  
   Create a fresh detector, populate it with data, and compute the best cycle once.

2. **Repeated updates + full recompute**  
   Reuse the same detector object, apply an update, then recompute from scratch.

3. **Repeated updates + incremental recompute**  
   Reuse the same detector object, apply a two-leg book update, then use the incremental path.

4. **Repeated updates + single-leg update**  
   Reuse the same detector object, apply a one-edge update, then use the single-edge incremental path.

### Results

| backend   | cold build + single compute | repeated updates + full recompute | repeated updates + incremental recompute | repeated updates + single-leg update |
|----------:|----------------------------:|----------------------------------:|-----------------------------------------:|-------------------------------------:|
| networkx  |                   88429.978 |                         87251.911 |                                87252.850 |                             87296.218 |
| generic   |                    2736.103 |                            89.019 |                                    1.185 |                                 0.643 |
| sse       |                    2677.078 |                            36.746 |                                    0.709 |                                 0.460 |
| avx       |                    2659.697 |                            20.547 |                                    0.653 |                                 0.451 |
| avx2      |                    2654.793 |                            18.720 |                                    0.647 |                                 0.441 |
| avx512    |                    2630.359 |                            12.861 |                                    0.641 |                                 0.445 |

Each cell is the median of seven CPU-pinned recorded runs after one discarded
warm-up. The NetworkX 3.6.1 baseline rebuilds or updates a `DiGraph` and
enumerates exact bounded simple cycles with `simple_cycles(length_bound=3)` on
every pass.

AVX-512 uses a specialized scan policy: padded dense weight rows remove the
scalar tail, while online threshold tightening folds the running best weight
into the broadcast prefix. Full recompute is an important workload, and on this
machine AVX-512 reduces it from **18.720 µs** with AVX2 to **12.861 µs**, a
31% latency reduction. The two-leg and single-leg results differ by about 1%
between AVX2 and AVX-512, which is benchmark noise at this scale. Auto-dispatch
therefore selects AVX-512 whenever it is compiled and supported; AVX2 remains
the next x86 fallback.

### What this means

The headline result is simple:

- the C++ backends are dramatically faster than a Python/NetworkX baseline
- incremental recompute is where the library really shines
- SIMD helps, especially once the detector is already built and hot

For the streaming use case, the important columns are the last three, not the first one.

Even the generic scalar backend is already very fast once the object is warm:

- **89.019 µs** for repeated full recompute
- **1.185 µs** for repeated two-leg incremental recompute
- **0.643 µs** for repeated one-leg incremental recompute

The SIMD backends push that further:

- **AVX-512** reached **12.861 µs** for repeated full recompute
- **AVX-512** reached **0.641 µs** for repeated two-leg incremental recompute
- **AVX-512** reached **0.445 µs** for repeated one-leg incremental recompute

### Why this benchmark matters

This library is intended for **streaming market-data workloads with tight latency budgets**.

That means the interesting question is usually not:

> how fast can I solve one giant graph problem once?

It is:

> how fast can I keep up with a stream of updates and repeatedly return the best cycle?

That is why NegCycle provides both:

- a full search API
- incremental update APIs for one-edge and two-leg book updates

And that is also why the incremental numbers matter most.

### Caveats

These numbers are focused only on:

- real market data
- 1,206 bid/ask currency-pair rows
- 3-currency cycles only
- a 4-vCPU KVM guest pinned to one logical CPU

Longer cycles, different pair universes, different update distributions, and
bare-metal CPUs will produce different results. The benchmark can be reproduced
with:

```bash
taskset -c 0 python benchmarks/headline_benchmark.py benchmark_data.csv
```

The result still shows the shape of the project clearly: NegCycle is aimed at
**fast repeated arbitrage detection under a live stream of changing prices**.

## Building without Poetry isolation

`poetry build` creates a temporary isolated build environment and tries to
download the backend declared in `[build-system].requires`. For this project,
that means fetching `scikit-build-core`, `nanobind`, `cmake`, and `ninja`
before the native build even starts.

If you are offline, behind a restricted index, or debugging local native build
issues, use the project environment directly instead:

```bash
poetry install --with dev
bash scripts/build-wheel-no-isolation.sh
```

That script runs:

```bash
poetry run python -m build --wheel --no-isolation
```

For editable installs:

```bash
poetry install --with dev
poetry run pip install --no-build-isolation -ve .
```

If you still want to use `poetry build`, make sure the machine can reach your
package index for the build backend dependencies.

## Files to edit first

- `src/negcycle/__init__.py`
- `src/cpp/module.cpp`
- the specific backend files under `src/cpp/backends/`
