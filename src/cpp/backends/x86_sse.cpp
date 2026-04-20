#include <stdexcept>

#include "common/detector_base.h"
#include "common/detector_bindings.h"
#include <immintrin.h>

namespace negcycle {

class SseArbitrageDetector final : public ArbitrageDetectorBase {
public:
  [[nodiscard]] std::optional<Cycle> find_best_arbitrage(int max_cycle_length) override;
  [[nodiscard]] std::optional<Cycle> add_quote_and_find_best_arbitrage(
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length) override;
  [[nodiscard]] std::optional<Cycle> add_book_and_find_best_arbitrage(
      std::string_view base,
      std::string_view quote,
      float bid,
      float ask,
      float fee_bps,
      int max_cycle_length) override;
  // [[nodiscard]] std::optional<Cycle> find_best_cycle_through_edge(
  //     int from,
  //     int to,
  //     int max_cycle_length) const override;
};
  
std::optional<Cycle>
SseArbitrageDetector::find_best_arbitrage(int max_cycle_length) {
  if (max_cycle_length < 2 || n() < 2) {
    cached_best_.reset();
    cached_max_cycle_length_ = max_cycle_length;
    cache_valid_ = true;
    return std::nullopt;
  }

  // Same hybrid policy as the SSE versions: SIMD for full recompute only
  // on the cycle lengths you benchmarked.
  if (max_cycle_length < 3 || max_cycle_length > 5) {
    SearchState state(max_cycle_length);
    state.visited.assign(static_cast<std::size_t>(n()), 0);
    state.path.reserve(static_cast<std::size_t>(max_cycle_length) + 1);

    for (int start = 0; start < n(); ++start) {
      std::fill(state.visited.begin(), state.visited.end(), static_cast<unsigned char>(0));
      state.path.clear();
      state.path.push_back(start);
      state.visited[static_cast<std::size_t>(start)] = 1;
      dfs_from_start(start, start, 0, 0.0f, 1.0f, state);
    }

    cached_best_ = state.best;
    cached_max_cycle_length_ = max_cycle_length;
    cache_valid_ = true;
    return cached_best_;
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
      } else if (
          len == best_len &&
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

  const __m128 zero_v = _mm_setzero_ps();
  alignas(16) float totals[4];

  for (int start = 0; start < N; ++start) {
    const float* close_to_start =
        WT.data() + static_cast<std::size_t>(start) * stride;
    const float* row_start =
        W.data() + static_cast<std::size_t>(start) * stride;
    const auto& adj_start = outgoing_[static_cast<std::size_t>(start)];

    for (int a : adj_start) {
      if (a <= start) {
        continue; // canonicalization: start must be the minimum vertex id
      }

      const float w_sa = row_start[static_cast<std::size_t>(a)];

      const float total2 = w_sa + close_to_start[static_cast<std::size_t>(a)];
      if (total2 < 0.0f) {
        const int path[3] = {start, a, start};
        record_candidate(total2, 2, path);
      }

      const float* row_a = W.data() + static_cast<std::size_t>(a) * stride;

      if (max_cycle_length >= 3) {
        const __m128 prefix_v = _mm_set1_ps(w_sa);
        const __m128 a_v = _mm_set1_ps(static_cast<float>(a));

        int j = start + 1;
        for (; j + 4 <= N; j += 4) {
          const __m128 idx_v =
              _mm_loadu_ps(index_f.data() + static_cast<std::size_t>(j));
          const __m128 valid_v = _mm_cmpneq_ps(idx_v, a_v);

          const __m128 total_v = _mm_add_ps(
              prefix_v,
              _mm_add_ps(
                  _mm_loadu_ps(row_a + static_cast<std::size_t>(j)),
                  _mm_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

          const int mask = _mm_movemask_ps(
              _mm_and_ps(valid_v, _mm_cmplt_ps(total_v, zero_v)));

          if (mask != 0) {
            _mm_storeu_ps(totals, total_v);
            for (int lane = 0; lane < 4; ++lane) {
              if ((mask & (1 << lane)) == 0) {
                continue;
              }
              const int b = j + lane;
              const int path[4] = {start, a, b, start};
              record_candidate(totals[lane], 3, path);
            }
          }
        }

        for (; j < N; ++j) {
          if (j == a) {
            continue;
          }
          const float total3 =
              w_sa +
              row_a[static_cast<std::size_t>(j)] +
              close_to_start[static_cast<std::size_t>(j)];
          if (total3 < 0.0f) {
            const int path[4] = {start, a, j, start};
            record_candidate(total3, 3, path);
          }
        }
      }

      if (max_cycle_length >= 4) {
        const auto& adj_a = outgoing_[static_cast<std::size_t>(a)];
        for (int b : adj_a) {
          if (b <= start || b == a) {
            continue;
          }

          const float prefix2 = w_sa + row_a[static_cast<std::size_t>(b)];
          const float* row_b = W.data() + static_cast<std::size_t>(b) * stride;
          const __m128 prefix_v = _mm_set1_ps(prefix2);
          const __m128 a_v = _mm_set1_ps(static_cast<float>(a));
          const __m128 b_v = _mm_set1_ps(static_cast<float>(b));

          int j = start + 1;
          for (; j + 4 <= N; j += 4) {
            const __m128 idx_v =
                _mm_loadu_ps(index_f.data() + static_cast<std::size_t>(j));

            __m128 valid_v = _mm_cmpneq_ps(idx_v, a_v);
            valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, b_v));

            const __m128 total_v = _mm_add_ps(
                prefix_v,
                _mm_add_ps(
                    _mm_loadu_ps(row_b + static_cast<std::size_t>(j)),
                    _mm_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

            const int mask = _mm_movemask_ps(
                _mm_and_ps(valid_v, _mm_cmplt_ps(total_v, zero_v)));

            if (mask != 0) {
              _mm_storeu_ps(totals, total_v);
              for (int lane = 0; lane < 4; ++lane) {
                if ((mask & (1 << lane)) == 0) {
                  continue;
                }
                const int c = j + lane;
                const int path[5] = {start, a, b, c, start};
                record_candidate(totals[lane], 4, path);
              }
            }
          }

          for (; j < N; ++j) {
            if (j == a || j == b) {
              continue;
            }
            const float total4 =
                prefix2 +
                row_b[static_cast<std::size_t>(j)] +
                close_to_start[static_cast<std::size_t>(j)];
            if (total4 < 0.0f) {
              const int path[5] = {start, a, b, j, start};
              record_candidate(total4, 4, path);
            }
          }
        }
      }

      if (max_cycle_length >= 5) {
        const auto& adj_a = outgoing_[static_cast<std::size_t>(a)];
        for (int b : adj_a) {
          if (b <= start || b == a) {
            continue;
          }

          const float prefix2 = w_sa + row_a[static_cast<std::size_t>(b)];
          const auto& adj_b = outgoing_[static_cast<std::size_t>(b)];

          for (int c : adj_b) {
            if (c <= start || c == a || c == b) {
              continue;
            }

            const float prefix3 =
                prefix2 +
                W[static_cast<std::size_t>(b) * stride +
                  static_cast<std::size_t>(c)];
            const float* row_c =
                W.data() + static_cast<std::size_t>(c) * stride;

            const __m128 prefix_v = _mm_set1_ps(prefix3);
            const __m128 a_v = _mm_set1_ps(static_cast<float>(a));
            const __m128 b_v = _mm_set1_ps(static_cast<float>(b));
            const __m128 c_v = _mm_set1_ps(static_cast<float>(c));

            int j = start + 1;
            for (; j + 4 <= N; j += 4) {
              const __m128 idx_v =
                  _mm_loadu_ps(index_f.data() + static_cast<std::size_t>(j));

              __m128 valid_v = _mm_cmpneq_ps(idx_v, a_v);
              valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, b_v));
              valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, c_v));

              const __m128 total_v = _mm_add_ps(
                  prefix_v,
                  _mm_add_ps(
                      _mm_loadu_ps(row_c + static_cast<std::size_t>(j)),
                      _mm_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

              const int mask = _mm_movemask_ps(
                  _mm_and_ps(valid_v, _mm_cmplt_ps(total_v, zero_v)));

              if (mask != 0) {
                _mm_storeu_ps(totals, total_v);
                for (int lane = 0; lane < 4; ++lane) {
                  if ((mask & (1 << lane)) == 0) {
                    continue;
                  }
                  const int d = j + lane;
                  const int path[6] = {start, a, b, c, d, start};
                  record_candidate(totals[lane], 5, path);
                }
              }
            }

            for (; j < N; ++j) {
              if (j == a || j == b || j == c) {
                continue;
              }
              const float total5 =
                  prefix3 +
                  row_c[static_cast<std::size_t>(j)] +
                  close_to_start[static_cast<std::size_t>(j)];
              if (total5 < 0.0f) {
                const int path[6] = {start, a, b, c, j, start};
                record_candidate(total5, 5, path);
              }
            }
          }
        }
      }
    }
  }

  if (!have_best) {
    cached_best_.reset();
  } else {
    std::vector<int> best_path(static_cast<std::size_t>(best_len) + 1);
    for (int i = 0; i <= best_len; ++i) {
      best_path[static_cast<std::size_t>(i)] = best_vertices[i];
    }

    float gain_factor = 1.0f;
    for (int i = 0; i < best_len; ++i) {
      gain_factor *= cell(best_vertices[i], best_vertices[i + 1]).net_rate;
    }

    cached_best_ = materialize_cycle(best_path, best_weight, gain_factor);
  }

  cached_max_cycle_length_ = max_cycle_length;
  cache_valid_ = true;
  return cached_best_;
}
  
std::optional<Cycle>
SseArbitrageDetector::add_quote_and_find_best_arbitrage(std::string_view from,
                                                     std::string_view to,
                                                     float executable_rate,
                                                     float fee_bps,
                                                     int max_cycle_length) {
  UpsertResult update = upsert_quote(from, to, executable_rate, fee_bps);

  if (max_cycle_length < 2 || n() < 2) {
    cached_best_.reset();
    cached_max_cycle_length_ = max_cycle_length;
    cache_valid_ = true;
    return std::nullopt;
  }

  if (!cache_valid_ || cached_max_cycle_length_ != max_cycle_length || update.new_currency) {
    return find_best_arbitrage(max_cycle_length);
  }

  const bool improved =
      update.new_edge || update.new_weight < update.old_weight - kCompareEpsilon;
  const bool worsened =
      update.existed && update.new_weight > update.old_weight + kCompareEpsilon;
  const bool cached_uses_edge =
      cached_best_.has_value() &&
      cycle_uses_edge(*cached_best_, update.from, update.to);

  auto scalar_forced_edge = [&]() -> std::optional<Cycle> {
    return find_best_cycle_through_edge(update.from, update.to, max_cycle_length);
  };

  auto sse_forced_edge_len5 = [&]() -> std::optional<Cycle> {
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

    const int start = update.from;
    const int second = update.to;

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
        } else if (
            len == best_len &&
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

    const float* close_to_start =
        WT.data() + static_cast<std::size_t>(start) * stride;

    const float total2 =
        first_w + close_to_start[static_cast<std::size_t>(second)];
    if (total2 < 0.0f) {
      const int path[3] = {start, second, start};
      record_candidate(total2, 2, path);
    }

    const __m128 zero_v = _mm_setzero_ps();
    alignas(16) float totals[4];
    const float* row_second =
        W.data() + static_cast<std::size_t>(second) * stride;

    if (max_cycle_length >= 3) {
      const __m128 prefix_v = _mm_set1_ps(first_w);
      const __m128 start_v = _mm_set1_ps(static_cast<float>(start));
      const __m128 second_v = _mm_set1_ps(static_cast<float>(second));

      int j = 0;
      for (; j + 4 <= N; j += 4) {
        const __m128 idx_v =
            _mm_loadu_ps(index_f.data() + static_cast<std::size_t>(j));

        __m128 valid_v = _mm_cmpneq_ps(idx_v, start_v);
        valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, second_v));

        const __m128 total_v = _mm_add_ps(
            prefix_v,
            _mm_add_ps(
                _mm_loadu_ps(row_second + static_cast<std::size_t>(j)),
                _mm_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

        const int mask = _mm_movemask_ps(
            _mm_and_ps(valid_v, _mm_cmplt_ps(total_v, zero_v)));

        if (mask != 0) {
          _mm_storeu_ps(totals, total_v);
          for (int lane = 0; lane < 4; ++lane) {
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
            first_w +
            row_second[static_cast<std::size_t>(j)] +
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
        const __m128 prefix_v = _mm_set1_ps(prefix2);
        const __m128 start_v = _mm_set1_ps(static_cast<float>(start));
        const __m128 second_v = _mm_set1_ps(static_cast<float>(second));
        const __m128 b_v = _mm_set1_ps(static_cast<float>(b));

        int j = 0;
        for (; j + 4 <= N; j += 4) {
          const __m128 idx_v =
              _mm_loadu_ps(index_f.data() + static_cast<std::size_t>(j));

          __m128 valid_v = _mm_cmpneq_ps(idx_v, start_v);
          valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, second_v));
          valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, b_v));

          const __m128 total_v = _mm_add_ps(
              prefix_v,
              _mm_add_ps(
                  _mm_loadu_ps(row_b + static_cast<std::size_t>(j)),
                  _mm_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

          const int mask = _mm_movemask_ps(
              _mm_and_ps(valid_v, _mm_cmplt_ps(total_v, zero_v)));

          if (mask != 0) {
            _mm_storeu_ps(totals, total_v);
            for (int lane = 0; lane < 4; ++lane) {
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
              prefix2 +
              row_b[static_cast<std::size_t>(j)] +
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
              prefix2 +
              W[static_cast<std::size_t>(b) * stride +
                static_cast<std::size_t>(c)];
          const float* row_c =
              W.data() + static_cast<std::size_t>(c) * stride;

          const __m128 prefix_v = _mm_set1_ps(prefix3);
          const __m128 start_v = _mm_set1_ps(static_cast<float>(start));
          const __m128 second_v = _mm_set1_ps(static_cast<float>(second));
          const __m128 b_v = _mm_set1_ps(static_cast<float>(b));
          const __m128 c_v = _mm_set1_ps(static_cast<float>(c));

          int j = 0;
          for (; j + 4 <= N; j += 4) {
            const __m128 idx_v =
                _mm_loadu_ps(index_f.data() + static_cast<std::size_t>(j));

            __m128 valid_v = _mm_cmpneq_ps(idx_v, start_v);
            valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, second_v));
            valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, b_v));
            valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, c_v));

            const __m128 total_v = _mm_add_ps(
                prefix_v,
                _mm_add_ps(
                    _mm_loadu_ps(row_c + static_cast<std::size_t>(j)),
                    _mm_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

            const int mask = _mm_movemask_ps(
                _mm_and_ps(valid_v, _mm_cmplt_ps(total_v, zero_v)));

            if (mask != 0) {
              _mm_storeu_ps(totals, total_v);
              for (int lane = 0; lane < 4; ++lane) {
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
                prefix3 +
                row_c[static_cast<std::size_t>(j)] +
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
  };

  if (improved) {
    std::optional<Cycle> through_edge =
        (max_cycle_length == 5) ? sse_forced_edge_len5()
                                : scalar_forced_edge();

    std::optional<Cycle> result;
    if (cached_best_ && !cached_uses_edge) {
      result = better_optional(cached_best_, through_edge);
    } else {
      result = through_edge;
    }

    cached_best_ = result;
    cached_max_cycle_length_ = max_cycle_length;
    cache_valid_ = true;
    return cached_best_;
  }

  if (worsened) {
    if (!cached_uses_edge) {
      return cached_best_;
    }
    return find_best_arbitrage(max_cycle_length);
  }

  return cached_best_;
}

std::optional<Cycle>
SseArbitrageDetector::add_book_and_find_best_arbitrage(std::string_view base,
                                                    std::string_view quote,
                                                    float bid,
                                                    float ask,
                                                    float fee_bps,
                                                    int max_cycle_length) {
  if (!(bid > 0.0f) || !(ask > 0.0f) || bid > ask) {
    throw std::invalid_argument("invalid bid/ask");
  }
  if (!(fee_bps >= 0.0f) || fee_bps >= 10000.0f) {
    throw std::invalid_argument("fee_bps must be in [0, 10000)");
  }

  const float reverse_rate = 1.0f / ask;

  bool forward_same = false;
  bool reverse_same = false;

  const auto it_base = id_by_code_.find(std::string(base));
  const auto it_quote = id_by_code_.find(std::string(quote));

  if (it_base != id_by_code_.end() && it_quote != id_by_code_.end()) {
    const int u = it_base->second;
    const int v = it_quote->second;

    const QuoteCell& forward = cell(u, v);
    const QuoteCell& reverse = cell(v, u);

    if (forward.exists &&
        std::fabs(forward.gross_rate - bid) <= kCompareEpsilon &&
        std::fabs(forward.fee_bps - fee_bps) <= kCompareEpsilon) {
      forward_same = true;
    }

    if (reverse.exists &&
        std::fabs(reverse.gross_rate - reverse_rate) <= kCompareEpsilon &&
        std::fabs(reverse.fee_bps - fee_bps) <= kCompareEpsilon) {
      reverse_same = true;
    }
  }

  if (forward_same && reverse_same) {
    if (cache_valid_ && cached_max_cycle_length_ == max_cycle_length) {
      return cached_best_;
    }
    return find_best_arbitrage(max_cycle_length);
  }

  if (forward_same && !reverse_same) {
    return add_quote_and_find_best_arbitrage(quote,
                                             base,
                                             reverse_rate,
                                             fee_bps,
                                             max_cycle_length);
  }

  if (!forward_same && reverse_same) {
    return add_quote_and_find_best_arbitrage(base,
                                             quote,
                                             bid,
                                             fee_bps,
                                             max_cycle_length);
  }

  UpsertResult forward = upsert_quote(base, quote, bid, fee_bps);
  UpsertResult reverse = upsert_quote(quote, base, reverse_rate, fee_bps);

  if (max_cycle_length < 2 || n() < 2) {
    cached_best_.reset();
    cached_max_cycle_length_ = max_cycle_length;
    cache_valid_ = true;
    return std::nullopt;
  }

  if (!cache_valid_ || cached_max_cycle_length_ != max_cycle_length ||
      forward.new_currency || reverse.new_currency) {
    return find_best_arbitrage(max_cycle_length);
  }

  const bool forward_improved =
      forward.new_edge || forward.new_weight < forward.old_weight - kCompareEpsilon;
  const bool forward_worsened =
      forward.existed && forward.new_weight > forward.old_weight + kCompareEpsilon;

  const bool reverse_improved =
      reverse.new_edge || reverse.new_weight < reverse.old_weight - kCompareEpsilon;
  const bool reverse_worsened =
      reverse.existed && reverse.new_weight > reverse.old_weight + kCompareEpsilon;

  const bool cached_uses_forward =
      cached_best_.has_value() &&
      cycle_uses_edge(*cached_best_, forward.from, forward.to);

  const bool cached_uses_reverse =
      cached_best_.has_value() &&
      cycle_uses_edge(*cached_best_, reverse.from, reverse.to);

  if ((forward_worsened && cached_uses_forward) ||
      (reverse_worsened && cached_uses_reverse)) {
    return find_best_arbitrage(max_cycle_length);
  }

  if (!forward_improved && !reverse_improved) {
    return cached_best_;
  }

  std::optional<Cycle> result;

  const bool cached_uses_improved_edge =
      (forward_improved && cached_uses_forward) ||
      (reverse_improved && cached_uses_reverse);

  if (cached_best_ && !cached_uses_improved_edge) {
    result = cached_best_;
  }

  if (max_cycle_length != 5) {
    if (forward_improved) {
      result = better_optional(
          std::move(result),
          find_best_cycle_through_edge(forward.from, forward.to, max_cycle_length));
    }

    if (reverse_improved) {
      result = better_optional(
          std::move(result),
          find_best_cycle_through_edge(reverse.from, reverse.to, max_cycle_length));
    }

    cached_best_ = result;
    cached_max_cycle_length_ = max_cycle_length;
    cache_valid_ = true;
    return cached_best_;
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

  auto find_best_through_edge_sse = [&](int start, int second)
      -> std::optional<Cycle> {
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
        } else if (
            len == best_len &&
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

    const float* close_to_start =
        WT.data() + static_cast<std::size_t>(start) * stride;

    const float total2 =
        first_w + close_to_start[static_cast<std::size_t>(second)];
    if (total2 < 0.0f) {
      const int path[3] = {start, second, start};
      record_candidate(total2, 2, path);
    }

    const __m128 zero_v = _mm_setzero_ps();
    alignas(16) float totals[4];
    const float* row_second =
        W.data() + static_cast<std::size_t>(second) * stride;

    if (max_cycle_length >= 3) {
      const __m128 prefix_v = _mm_set1_ps(first_w);
      const __m128 start_v = _mm_set1_ps(static_cast<float>(start));
      const __m128 second_v = _mm_set1_ps(static_cast<float>(second));

      int j = 0;
      for (; j + 4 <= N; j += 4) {
        const __m128 idx_v =
            _mm_loadu_ps(index_f.data() + static_cast<std::size_t>(j));

        __m128 valid_v = _mm_cmpneq_ps(idx_v, start_v);
        valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, second_v));

        const __m128 total_v = _mm_add_ps(
            prefix_v,
            _mm_add_ps(
                _mm_loadu_ps(row_second + static_cast<std::size_t>(j)),
                _mm_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

        const int mask = _mm_movemask_ps(
            _mm_and_ps(valid_v, _mm_cmplt_ps(total_v, zero_v)));

        if (mask != 0) {
          _mm_storeu_ps(totals, total_v);
          for (int lane = 0; lane < 4; ++lane) {
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
            first_w +
            row_second[static_cast<std::size_t>(j)] +
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
        const __m128 prefix_v = _mm_set1_ps(prefix2);
        const __m128 start_v = _mm_set1_ps(static_cast<float>(start));
        const __m128 second_v = _mm_set1_ps(static_cast<float>(second));
        const __m128 b_v = _mm_set1_ps(static_cast<float>(b));

        int j = 0;
        for (; j + 4 <= N; j += 4) {
          const __m128 idx_v =
              _mm_loadu_ps(index_f.data() + static_cast<std::size_t>(j));

          __m128 valid_v = _mm_cmpneq_ps(idx_v, start_v);
          valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, second_v));
          valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, b_v));

          const __m128 total_v = _mm_add_ps(
              prefix_v,
              _mm_add_ps(
                  _mm_loadu_ps(row_b + static_cast<std::size_t>(j)),
                  _mm_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

          const int mask = _mm_movemask_ps(
              _mm_and_ps(valid_v, _mm_cmplt_ps(total_v, zero_v)));

          if (mask != 0) {
            _mm_storeu_ps(totals, total_v);
            for (int lane = 0; lane < 4; ++lane) {
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
              prefix2 +
              row_b[static_cast<std::size_t>(j)] +
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
              prefix2 +
              W[static_cast<std::size_t>(b) * stride +
                static_cast<std::size_t>(c)];
          const float* row_c =
              W.data() + static_cast<std::size_t>(c) * stride;

          const __m128 prefix_v = _mm_set1_ps(prefix3);
          const __m128 start_v = _mm_set1_ps(static_cast<float>(start));
          const __m128 second_v = _mm_set1_ps(static_cast<float>(second));
          const __m128 b_v = _mm_set1_ps(static_cast<float>(b));
          const __m128 c_v = _mm_set1_ps(static_cast<float>(c));

          int j = 0;
          for (; j + 4 <= N; j += 4) {
            const __m128 idx_v =
                _mm_loadu_ps(index_f.data() + static_cast<std::size_t>(j));

            __m128 valid_v = _mm_cmpneq_ps(idx_v, start_v);
            valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, second_v));
            valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, b_v));
            valid_v = _mm_and_ps(valid_v, _mm_cmpneq_ps(idx_v, c_v));

            const __m128 total_v = _mm_add_ps(
                prefix_v,
                _mm_add_ps(
                    _mm_loadu_ps(row_c + static_cast<std::size_t>(j)),
                    _mm_loadu_ps(close_to_start + static_cast<std::size_t>(j))));

            const int mask = _mm_movemask_ps(
                _mm_and_ps(valid_v, _mm_cmplt_ps(total_v, zero_v)));

            if (mask != 0) {
              _mm_storeu_ps(totals, total_v);
              for (int lane = 0; lane < 4; ++lane) {
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
                prefix3 +
                row_c[static_cast<std::size_t>(j)] +
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
  };

  if (forward_improved) {
    result = better_optional(
        std::move(result),
        find_best_through_edge_sse(forward.from, forward.to));
  }

  if (reverse_improved) {
    result = better_optional(
        std::move(result),
        find_best_through_edge_sse(reverse.from, reverse.to));
  }

  cached_best_ = result;
  cached_max_cycle_length_ = max_cycle_length;
  cache_valid_ = true;
  return cached_best_;
}

  
  
} // namespace negcycle

namespace nb = nanobind;

NB_MODULE(_sse, m) {
  negcycle::bind_detector_module<negcycle::SseArbitrageDetector>(
      m,
      "_SseArbitrageDetector",
      "SSE bounded simple-cycle arbitrage detector");
}
