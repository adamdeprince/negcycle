# Benchmarks

The Intel benchmarks were rerun on 2026-07-29 against NegCycle commit
`37226cb`. The macOS and Loongson results remain the 2026-05-13 measurements
against the same compact market-data input.

`benchmark_data.csv` contains 1,206 data rows (1,207 lines including its
header) and has SHA-256
`ef8b149ba0a312bddcdf0ec1b77be47cd8abbe4898292fa540945e156aa4b6bf`.

## Machines

| Machine | CPU | OS | Compiler | Python | Commit | Run date |
| --- | --- | --- | --- | --- | --- | --- |
| Intel | Intel Xeon 6975P-C, 4 vCPU KVM guest | Ubuntu 26.04 / Linux 7.0.0-1008-aws | g++ 16.1.0 (GCC) | Python 3.14.4 | `37226cb` | 2026-07-29 |
| macOS | Apple M4 Max | macOS 15.3.1 / Darwin 24.3.0 | Apple clang++ 17.0.0 (clang-1700.0.13.5) | Python 3.13.3 | `fed28e2` | 2026-05-13 |
| Loongson | Loongson-3A6000 | Kylin V10 SP1 / Linux 5.4.18-110-generic | g++ 15.2.0 (GCC) | Python 3.13.13 | `c6ac527` | 2026-05-13 |

## Headline streaming results

These are medians of seven recorded runs after one discarded warm-up, pinned
to logical CPU 0. Values are **microseconds per pass**; lower is better.
NetworkX 3.6.1 enumerates exact bounded simple cycles with
`simple_cycles(length_bound=3)` on every pass.

| Backend | repeated single-leg update | repeated two-leg incremental | repeated full recompute | cold build + single compute |
| --- | ---: | ---: | ---: | ---: |
| networkx | 87296.218 | 87252.850 | 87251.911 | 88429.978 |
| generic | 0.643 | 1.185 | 89.019 | 2736.103 |
| sse | 0.460 | 0.709 | 36.746 | 2677.078 |
| avx | 0.451 | 0.653 | 20.547 | 2659.697 |
| avx2 | 0.441 | 0.647 | 18.720 | 2654.793 |
| avx512 | 0.445 | 0.641 | 12.861 | 2630.359 |

The main result is algorithmic: the generic backend drops from **89.019 µs**
for full recompute to **1.185 µs** for two-leg incremental recompute. AVX-512
then improves the incremental result to **0.641 µs**, about 1.85× faster than
generic. AVX2 narrowly leads the smallest single-leg workload at **0.441 µs**.

The median rows and full provenance are in
[headline_benchmarks.csv](headline_benchmarks.csv).

## Separate backend-throughput results

This suite reports **calls/sec** and speedup against that machine's generic
backend. It is a separate measurement from the headline µs/pass table. The
2026-07-29 Intel figures are medians from seven runs with `--passes 20`, after
one warm-up, pinned to logical CPU 0. Speedups are calculated from the median
calls/sec values. The older macOS and Loongson figures use the original
`--passes 4` run.

### Intel: Xeon 6975P-C

| Backend | add quote best | add quote all | add book best | add book all |
| --- | ---: | ---: | ---: | ---: |
| generic | 2,812,169 calls/s | 6,463 calls/s | 841,367 calls/s | 4,927 calls/s |
| x86_sse | 1.02x | 1.40x | 1.66x | 1.41x |
| x86_avx | 1.03x | 1.59x | 1.82x | 1.55x |
| x86_avx2 | 1.01x | 1.59x | 1.84x | 1.57x |
| x86_avx512 | 1.02x | 1.75x | 1.82x | 1.70x |

### macOS: Apple M4 Max

| Backend | add quote best | add quote all | add book best | add book all |
| --- | ---: | ---: | ---: | ---: |
| generic | 2,851,892 calls/s | 14,634 calls/s | 1,131,328 calls/s | 10,070 calls/s |
| macos_arm64_neon | 1.23x | 1.54x | 1.78x | 1.59x |

### Loongson: Loongson-3A6000

| Backend | add quote best | add quote all | add book best | add book all |
| --- | ---: | ---: | ---: | ---: |
| generic | 764,124 calls/s | 4,327 calls/s | 318,830 calls/s | 2,988 calls/s |
| linux_loongarch64_lsx | 0.98x | 1.19x | 1.31x | 1.25x |
| linux_loongarch64_lasx | 0.88x | 1.33x | 1.32x | 1.38x |

## Notes

The x86 auto-dispatch target on the Intel benchmark machine is AVX2 even though
AVX-512 is available; AVX-512 remains an explicitly importable experimental
backend.

On the Intel machine, AVX-512 leads the all-cycle variants at 1.75× for quote
updates and 1.70× for book updates. AVX2 leads book-best at 1.84×. The
quote-best workload is too small for wide SIMD to dominate: all x86 SIMD
results are within 3% of generic there.

On Loongson, LASX is faster than LSX for the all-cycle variants and roughly even
for book best, but measures **0.88×** against generic for quote best. The
quote-best workload is small enough that SIMD mask and update overhead
dominates at that size.

## Reproduction

After installing NegCycle and NetworkX in the benchmark environment:

```bash
taskset -c 0 python benchmarks/headline_benchmark.py benchmark_data.csv
taskset -c 0 python benchmarks/quick_simd_benchmark.py benchmark_data.csv --passes 20
```

Run the second command seven times and take the median of each backend/function
row. The full backend-throughput rows are in
[benchmarks.csv](benchmarks.csv).
