# Benchmarks

Benchmarks were run on 2026-05-13 with `benchmarks/quick_simd_benchmark.py`
against the same 1,207-row `benchmark_data.csv` input on each machine.
Each SIMD result is reported as speedup versus that machine's generic backend.

## Machines

| Machine | Host | CPU | OS | Compiler | Python | Commit |
| --- | --- | --- | --- | --- | --- | --- |
| x86 | `jane` | 11th Gen Intel Core i7-1195G7 @ 2.90GHz | Linux 6.17.0-22-generic | g++ 15.2.0 (Ubuntu 15.2.0-4ubuntu4) | Python 3.13.7 | `fed28e2` |
| macOS | `wopr` | Apple M4 Max | macOS 15.3.1 / Darwin 24.3.0 | Apple clang++ 17.0.0 (clang-1700.0.13.5) | Python 3.13.3 | `fed28e2` |
| Loongson | `loongson` | Loongson-3A6000 | Kylin V10 SP1 / Linux 5.4.18-110-generic | g++ 15.2.0 (GCC) | Python 3.13.13 | `c6ac527` |

## Results

### x86: Intel i7-1195G7

| Backend | add quote best | add quote all | add book best | add book all |
| --- | ---: | ---: | ---: | ---: |
| generic | 1,462,611 calls/s | 5,958 calls/s | 559,424 calls/s | 4,337 calls/s |
| x86_sse | 1.00x | 1.53x | 1.65x | 1.52x |
| x86_avx | 1.13x | 1.68x | 1.67x | 1.59x |
| x86_avx2 | 1.10x | 1.68x | 1.71x | 1.58x |
| x86_avx512 | 1.06x | 1.64x | 1.78x | 1.56x |

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

The x86 auto-dispatch target on this machine is AVX2 even though AVX512 is
available; AVX512 remains an explicitly importable experimental backend.

On Loongson, LASX is faster than LSX for the all-cycle variants and roughly even
for book best, but slower for quote best in this run. The quote-best workload is
small enough that SIMD mask and update overhead can dominate.

The full row-level data is in `benchmarks.csv`.
