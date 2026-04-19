#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/detector_base.h"
#include "common/detector_bindings.h"

namespace arbcycle {

class AvxArbitrageDetector final : public ArbitrageDetectorBase {
protected:
  [[nodiscard]] std::optional<Cycle> find_best_cycle_through_edge_optimized(
      int from,
      int to,
      int max_cycle_length) const override {
    if (max_cycle_length < 5) {
      return find_best_cycle_through_edge_scalar(from, to, max_cycle_length);
    }

    const int N = n();
    const std::size_t stride = static_cast<std::size_t>(N);
    const float inf = std::numeric_limits<float>::infinity();

    std::vector<float> W(stride * stride, inf);
    std::vector<float> WT(stride * stride, inf);
    std::vector<float> index_f(stride);

    for (int i = 0; i < N; ++i) {
      index_f[static_cast<std::size_t>(i)] = static_cast<float>(i);
      for (int j = 0; j < N; ++j) {
        const QuoteCell& q = cell(i, j);
        if (!q.exists) {
          continue;
        }
        const std::size_t ij =
            static_cast<std::size_t>(i) * stride + static_cast<std::size_t>(j);
        const std::size_t ji =
            static_cast<std::size_t>(j) * stride + static_cast<std::size_t>(i);
        W[ij] = q.weight;
        WT[ji] = q.weight;
      }
    }

    const int start = from;
    const int second = to;

    if (start == second) {
      return std::nullopt;
    }

    const float first_w =
        W[static_cast<std::size_t>(start) * stride + static_cast<std::size_t>(second)];
    if (!std::isfinite(first_w)) {
      return std::nullopt;
    }

    bool have_best = false;
    float best_weight = inf;
    int best_len = 0;
    int best_vertices[6] = {0, 0, 0, 0, 0, 0};

    auto record_candidate = [&](float total_weight, int len, const int* vertices) {
      if (!(total_weight < 0.0f)) {
        return;
      }

      bool better = false;
      if (!have_best) {
        better = true;
      } else if (total_weight < best_weight - kCompareEpsilon) {
        better = true;
      } else if (!(best_weight < total_weight - kCompareEpsilon)) {
        if (len < best_len) {
          better = true;
        } else if (len == best_len &&
                   std::lexicographical_compare(vertices,
                                                vertices + len + 1,
                                                best_vertices,
                                                best_vertices + best_len + 1)) {
          better = true;
        }
      }

      if (better) {
        have_best = true;
        best_weight = total_weight;
        best_len = len;
        for (int i = 0; i <= len; ++i) {
          best_vertices[i] = vertices[i];
        }
      }
    };

    const float* close_to_start = WT.data() + static_cast<std::size_t>(start) * stride;

    const float total2 = first_w + close_to_start[static_cast<std::size_t>(second)];
    if (total2 < 0.0f) {
      const int path[3] = {start, second, start};
      record_candidate(total2, 2, path);
    }

    const __m256 zero_v = _mm256_setzero_ps();
    alignas(32) float totals[8];
    const float* row_second = W.data() + static_cast<std::size_t>(second) * stride;

    if (max_cycle_length >= 3) {
      const __m256 prefix_v = _mm256_set1_ps(first_w);
      const __m256 start_v = _mm256_set1_ps(static_cast<float>(start));
      const __m256 second_v = _mm256_set1_ps(static_cast<float>(second));

      int j = 0;
      for (; j + 8 <= N; j += 8) {
        const __m256 idx_v =
            _mm256_loadu_ps(index_f.data() + static_cast<std::size_t>(j));

        __m256 valid_v = _mm256_cmp_ps(idx_v, start_v, _CMP_NEQ_OQ);
        valid_v = _mm256_and_ps(valid_v, _mm256_cmp_ps(idx_v, second_v, _CMP_NEQ_OQ));

        const __m256 total_v = _mm256_add_ps(
            prefix_v,
            _mm256_add_ps(
                _mm256_loadu_ps(row_second + static_cast<std::size_t>(j)),
                _mm256_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

        const int mask = _mm256_movemask_ps(
            _mm256_and_ps(valid_v, _mm256_cmp_ps(total_v, zero_v, _CMP_LT_OQ)));

        if (mask != 0) {
          _mm256_storeu_ps(totals, total_v);
          for (int lane = 0; lane < 8; ++lane) {
            if ((mask & (1 << lane)) == 0) {
              continue;
            }
            const int b = j + lane;
            const int path[4] = {start, second, b, start};
            record_candidate(totals[lane], 3, path);
          }
        }
      }

      for (; j < N; ++j) {
        if (j == start || j == second) {
          continue;
        }
        const float total3 =
            first_w + row_second[static_cast<std::size_t>(j)] +
            close_to_start[static_cast<std::size_t>(j)];
        if (total3 < 0.0f) {
          const int path[4] = {start, second, j, start};
          record_candidate(total3, 3, path);
        }
      }
    }

    if (max_cycle_length >= 4) {
      const auto& adj_second = outgoing_[static_cast<std::size_t>(second)];
      for (int b : adj_second) {
        if (b == start || b == second) {
          continue;
        }

        const float prefix2 = first_w + row_second[static_cast<std::size_t>(b)];
        const float* row_b = W.data() + static_cast<std::size_t>(b) * stride;
        const __m256 prefix_v = _mm256_set1_ps(prefix2);
        const __m256 start_v = _mm256_set1_ps(static_cast<float>(start));
        const __m256 second_v = _mm256_set1_ps(static_cast<float>(second));
        const __m256 b_v = _mm256_set1_ps(static_cast<float>(b));

        int j = 0;
        for (; j + 8 <= N; j += 8) {
          const __m256 idx_v =
              _mm256_loadu_ps(index_f.data() + static_cast<std::size_t>(j));

          __m256 valid_v = _mm256_cmp_ps(idx_v, start_v, _CMP_NEQ_OQ);
          valid_v = _mm256_and_ps(valid_v, _mm256_cmp_ps(idx_v, second_v, _CMP_NEQ_OQ));
          valid_v = _mm256_and_ps(valid_v, _mm256_cmp_ps(idx_v, b_v, _CMP_NEQ_OQ));

          const __m256 total_v = _mm256_add_ps(
              prefix_v,
              _mm256_add_ps(
                  _mm256_loadu_ps(row_b + static_cast<std::size_t>(j)),
                  _mm256_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

          const int mask = _mm256_movemask_ps(
              _mm256_and_ps(valid_v, _mm256_cmp_ps(total_v, zero_v, _CMP_LT_OQ)));

          if (mask != 0) {
            _mm256_storeu_ps(totals, total_v);
            for (int lane = 0; lane < 8; ++lane) {
              if ((mask & (1 << lane)) == 0) {
                continue;
              }
              const int c = j + lane;
              const int path[5] = {start, second, b, c, start};
              record_candidate(totals[lane], 4, path);
            }
          }
        }

        for (; j < N; ++j) {
          if (j == start || j == second || j == b) {
            continue;
          }
          const float total4 =
              prefix2 + row_b[static_cast<std::size_t>(j)] +
              close_to_start[static_cast<std::size_t>(j)];
          if (total4 < 0.0f) {
            const int path[5] = {start, second, b, j, start};
            record_candidate(total4, 4, path);
          }
        }
      }
    }

    if (max_cycle_length >= 5) {
      const auto& adj_second = outgoing_[static_cast<std::size_t>(second)];
      for (int b : adj_second) {
        if (b == start || b == second) {
          continue;
        }

        const float prefix2 = first_w + row_second[static_cast<std::size_t>(b)];
        const auto& adj_b = outgoing_[static_cast<std::size_t>(b)];

        for (int c : adj_b) {
          if (c == start || c == second || c == b) {
            continue;
          }

          const float prefix3 =
              prefix2 + W[static_cast<std::size_t>(b) * stride + static_cast<std::size_t>(c)];
          const float* row_c = W.data() + static_cast<std::size_t>(c) * stride;

          const __m256 prefix_v = _mm256_set1_ps(prefix3);
          const __m256 start_v = _mm256_set1_ps(static_cast<float>(start));
          const __m256 second_v = _mm256_set1_ps(static_cast<float>(second));
          const __m256 b_v = _mm256_set1_ps(static_cast<float>(b));
          const __m256 c_v = _mm256_set1_ps(static_cast<float>(c));

          int j = 0;
          for (; j + 8 <= N; j += 8) {
            const __m256 idx_v =
                _mm256_loadu_ps(index_f.data() + static_cast<std::size_t>(j));

            __m256 valid_v = _mm256_cmp_ps(idx_v, start_v, _CMP_NEQ_OQ);
            valid_v = _mm256_and_ps(valid_v, _mm256_cmp_ps(idx_v, second_v, _CMP_NEQ_OQ));
            valid_v = _mm256_and_ps(valid_v, _mm256_cmp_ps(idx_v, b_v, _CMP_NEQ_OQ));
            valid_v = _mm256_and_ps(valid_v, _mm256_cmp_ps(idx_v, c_v, _CMP_NEQ_OQ));

            const __m256 total_v = _mm256_add_ps(
                prefix_v,
                _mm256_add_ps(
                    _mm256_loadu_ps(row_c + static_cast<std::size_t>(j)),
                    _mm256_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

            const int mask = _mm256_movemask_ps(
                _mm256_and_ps(valid_v, _mm256_cmp_ps(total_v, zero_v, _CMP_LT_OQ)));

            if (mask != 0) {
              _mm256_storeu_ps(totals, total_v);
              for (int lane = 0; lane < 8; ++lane) {
                if ((mask & (1 << lane)) == 0) {
                  continue;
                }
                const int d = j + lane;
                const int path[6] = {start, second, b, c, d, start};
                record_candidate(totals[lane], 5, path);
              }
            }
          }

          for (; j < N; ++j) {
            if (j == start || j == second || j == b || j == c) {
              continue;
            }
            const float total5 =
                prefix3 + row_c[static_cast<std::size_t>(j)] +
                close_to_start[static_cast<std::size_t>(j)];
            if (total5 < 0.0f) {
              const int path[6] = {start, second, b, c, j, start};
              record_candidate(total5, 5, path);
            }
          }
        }
      }
    }

    if (!have_best) {
      return std::nullopt;
    }

    std::vector<int> path(static_cast<std::size_t>(best_len) + 1);
    for (int i = 0; i <= best_len; ++i) {
      path[static_cast<std::size_t>(i)] = best_vertices[i];
    }

    float gain_factor = 1.0f;
    for (int i = 0; i < best_len; ++i) {
      gain_factor *= cell(best_vertices[i], best_vertices[i + 1]).net_rate;
    }

    return materialize_cycle(path, best_weight, gain_factor);
  }
};

} // namespace arbcycle

namespace nb = nanobind;

NB_MODULE(_avx, m) {
  arbcycle::bind_detector_module<arbcycle::AvxArbitrageDetector>(
      m,
      "_AvxArbitrageDetector",
      "AVX-accelerated bounded simple-cycle arbitrage detector");
}
