#include <stdexcept>

#include "common/detector_base.h"
#include "common/detector_bindings.h"
#include <arm_neon.h>
#include <algorithm>

namespace arbcycle {

namespace {

inline int neon_movemask_u32(uint32x4_t mask) noexcept {
  alignas(16) uint32_t lanes[4];
  vst1q_u32(lanes, mask);

  int bits = 0;
  for (int lane = 0; lane < 4; ++lane) {
    bits |= static_cast<int>((lanes[lane] >> 31) << lane);
  }
  return bits;
}

} // namespace


class NeonArbitrageDetector final : public ArbitrageDetectorBase {
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
  [[nodiscard]] std::optional<Cycle> find_best_cycle_through_edge(
      int from,
      int to,
      int max_cycle_length) const override;
};
  
std::optional<Cycle>
NeonArbitrageDetector::find_best_arbitrage(int max_cycle_length) {
  if (max_cycle_length < 2 || n() < 2) {
    cached_best_.reset();
    cached_max_cycle_length_ = max_cycle_length;
    cache_valid_ = true;
    return std::nullopt;
  }

  // Keep the old exact DFS for longer cycles.
  if (max_cycle_length > 5) {
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

  // NEON wants contiguous float rows. Build dense weight matrix and its transpose.
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

  const float32x4_t zero_v = vdupq_n_f32(0.0f);
  alignas(16) float totals[4];

  for (int start = 0; start < N; ++start) {
    const float* close_to_start =
        WT.data() + static_cast<std::size_t>(start) * stride;
    const float* row_start =
        W.data() + static_cast<std::size_t>(start) * stride;
    const auto& adj_start = outgoing_[static_cast<std::size_t>(start)];

    for (int a : adj_start) {
      if (a <= start) {
        continue;  // canonicalization: start must be the minimum vertex id
      }

      const float w_sa = row_start[static_cast<std::size_t>(a)];

      const float total2 = w_sa + close_to_start[static_cast<std::size_t>(a)];
      if (total2 < 0.0f) {
        const int path[3] = {start, a, start};
        record_candidate(total2, 2, path);
      }

      const float* row_a = W.data() + static_cast<std::size_t>(a) * stride;

      if (max_cycle_length >= 3) {
        const float32x4_t prefix_v = vdupq_n_f32(w_sa);
        const float32x4_t a_v = vdupq_n_f32(static_cast<float>(a));

        int j = start + 1;
        for (; j + 4 <= N; j += 4) {
          const float32x4_t idx_v =
              vld1q_f32(index_f.data() + static_cast<std::size_t>(j));
          const uint32x4_t valid_v = vmvnq_u32(vceqq_f32(idx_v, a_v));

          const float32x4_t total_v = vaddq_f32(
              prefix_v,
              vaddq_f32(
                  vld1q_f32(row_a + static_cast<std::size_t>(j)),
                  vld1q_f32(close_to_start + static_cast<std::size_t>(j))));

          const int mask = neon_movemask_u32(vandq_u32(valid_v, vcltq_f32(total_v, zero_v)));

          if (mask != 0) {
            vst1q_f32(totals, total_v);
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
          const float32x4_t prefix_v = vdupq_n_f32(prefix2);
          const float32x4_t a_v = vdupq_n_f32(static_cast<float>(a));
          const float32x4_t b_v = vdupq_n_f32(static_cast<float>(b));

          int j = start + 1;
          for (; j + 4 <= N; j += 4) {
            const float32x4_t idx_v =
                vld1q_f32(index_f.data() + static_cast<std::size_t>(j));

            uint32x4_t valid_v = vmvnq_u32(vceqq_f32(idx_v, a_v));
            valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, b_v)));

            const float32x4_t total_v = vaddq_f32(
                prefix_v,
                vaddq_f32(
                    vld1q_f32(row_b + static_cast<std::size_t>(j)),
                    vld1q_f32(close_to_start + static_cast<std::size_t>(j))));

            const int mask = neon_movemask_u32(vandq_u32(valid_v, vcltq_f32(total_v, zero_v)));

            if (mask != 0) {
              vst1q_f32(totals, total_v);
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

            const float32x4_t prefix_v = vdupq_n_f32(prefix3);
            const float32x4_t a_v = vdupq_n_f32(static_cast<float>(a));
            const float32x4_t b_v = vdupq_n_f32(static_cast<float>(b));
            const float32x4_t c_v = vdupq_n_f32(static_cast<float>(c));

            int j = start + 1;
            for (; j + 4 <= N; j += 4) {
              const float32x4_t idx_v =
                  vld1q_f32(index_f.data() + static_cast<std::size_t>(j));

              uint32x4_t valid_v = vmvnq_u32(vceqq_f32(idx_v, a_v));
              valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, b_v)));
              valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, c_v)));

              const float32x4_t total_v = vaddq_f32(
                  prefix_v,
                  vaddq_f32(
                      vld1q_f32(row_c + static_cast<std::size_t>(j)),
                      vld1q_f32(close_to_start + static_cast<std::size_t>(j))));

              const int mask = neon_movemask_u32(vandq_u32(valid_v, vcltq_f32(total_v, zero_v)));

              if (mask != 0) {
                vst1q_f32(totals, total_v);
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
NeonArbitrageDetector::add_quote_and_find_best_arbitrage(
    std::string_view from,
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

  if (improved) {
    std::optional<Cycle> through_edge;

    if (max_cycle_length == 5) {
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

      auto find_best_through_edge_neon = [&](int start, int second)
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

        const float32x4_t zero_v = vdupq_n_f32(0.0f);
        alignas(16) float totals[4];

        if (max_cycle_length >= 3) {
          const float* row_second =
              W.data() + static_cast<std::size_t>(second) * stride;
          const float32x4_t prefix_v = vdupq_n_f32(first_w);
          const float32x4_t start_v = vdupq_n_f32(static_cast<float>(start));
          const float32x4_t second_v = vdupq_n_f32(static_cast<float>(second));

          int j = 0;
          for (; j + 4 <= N; j += 4) {
            const float32x4_t idx_v =
                vld1q_f32(index_f.data() + static_cast<std::size_t>(j));

            uint32x4_t valid_v = vmvnq_u32(vceqq_f32(idx_v, start_v));
            valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, second_v)));

            const float32x4_t total_v = vaddq_f32(
                prefix_v,
                vaddq_f32(
                    vld1q_f32(row_second + static_cast<std::size_t>(j)),
                    vld1q_f32(close_to_start + static_cast<std::size_t>(j))));

            const int mask = neon_movemask_u32(vandq_u32(valid_v, vcltq_f32(total_v, zero_v)));

            if (mask != 0) {
              vst1q_f32(totals, total_v);
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
          const float* row_second =
              W.data() + static_cast<std::size_t>(second) * stride;

          for (int b : adj_second) {
            if (b == start || b == second) {
              continue;
            }

            const float prefix2 = first_w + row_second[static_cast<std::size_t>(b)];
            const float* row_b =
                W.data() + static_cast<std::size_t>(b) * stride;
            const float32x4_t prefix_v = vdupq_n_f32(prefix2);
            const float32x4_t start_v = vdupq_n_f32(static_cast<float>(start));
            const float32x4_t second_v = vdupq_n_f32(static_cast<float>(second));
            const float32x4_t b_v = vdupq_n_f32(static_cast<float>(b));

            int j = 0;
            for (; j + 4 <= N; j += 4) {
              const float32x4_t idx_v =
                  vld1q_f32(index_f.data() + static_cast<std::size_t>(j));

              uint32x4_t valid_v = vmvnq_u32(vceqq_f32(idx_v, start_v));
              valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, second_v)));
              valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, b_v)));

              const float32x4_t total_v = vaddq_f32(
                  prefix_v,
                  vaddq_f32(
                      vld1q_f32(row_b + static_cast<std::size_t>(j)),
                      vld1q_f32(close_to_start + static_cast<std::size_t>(j))));

              const int mask = neon_movemask_u32(vandq_u32(valid_v, vcltq_f32(total_v, zero_v)));

              if (mask != 0) {
                vst1q_f32(totals, total_v);
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
          const float* row_second =
              W.data() + static_cast<std::size_t>(second) * stride;

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

              const float32x4_t prefix_v = vdupq_n_f32(prefix3);
              const float32x4_t start_v = vdupq_n_f32(static_cast<float>(start));
              const float32x4_t second_v = vdupq_n_f32(static_cast<float>(second));
              const float32x4_t b_v = vdupq_n_f32(static_cast<float>(b));
              const float32x4_t c_v = vdupq_n_f32(static_cast<float>(c));

              int j = 0;
              for (; j + 4 <= N; j += 4) {
                const float32x4_t idx_v =
                    vld1q_f32(index_f.data() + static_cast<std::size_t>(j));

                uint32x4_t valid_v = vmvnq_u32(vceqq_f32(idx_v, start_v));
                valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, second_v)));
                valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, b_v)));
                valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, c_v)));

                const float32x4_t total_v = vaddq_f32(
                    prefix_v,
                    vaddq_f32(
                        vld1q_f32(row_c + static_cast<std::size_t>(j)),
                        vld1q_f32(close_to_start + static_cast<std::size_t>(j))));

                const int mask = neon_movemask_u32(vandq_u32(valid_v, vcltq_f32(total_v, zero_v)));

                if (mask != 0) {
                  vst1q_f32(totals, total_v);
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

      through_edge = find_best_through_edge_neon(update.from, update.to);
    } else {
      through_edge = scalar_forced_edge();
    }

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
NeonArbitrageDetector::add_book_and_find_best_arbitrage(
    std::string_view base,
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

    const QuoteCell& forward = cell(u, v);  // base -> quote at bid
    const QuoteCell& reverse = cell(v, u);  // quote -> base at 1/ask

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

  auto find_best_through_edge_neon = [&](int start, int second)
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

    const float32x4_t zero_v = vdupq_n_f32(0.0f);
    alignas(16) float totals[4];

    if (max_cycle_length >= 3) {
      const float* row_second =
          W.data() + static_cast<std::size_t>(second) * stride;
      const float32x4_t prefix_v = vdupq_n_f32(first_w);
      const float32x4_t start_v = vdupq_n_f32(static_cast<float>(start));
      const float32x4_t second_v = vdupq_n_f32(static_cast<float>(second));

      int j = 0;
      for (; j + 4 <= N; j += 4) {
        const float32x4_t idx_v =
            vld1q_f32(index_f.data() + static_cast<std::size_t>(j));

        uint32x4_t valid_v = vmvnq_u32(vceqq_f32(idx_v, start_v));
        valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, second_v)));

        const float32x4_t total_v = vaddq_f32(
            prefix_v,
            vaddq_f32(
                vld1q_f32(row_second + static_cast<std::size_t>(j)),
                vld1q_f32(close_to_start + static_cast<std::size_t>(j))));

        const int mask = neon_movemask_u32(vandq_u32(valid_v, vcltq_f32(total_v, zero_v)));

        if (mask != 0) {
          vst1q_f32(totals, total_v);
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
      const float* row_second =
          W.data() + static_cast<std::size_t>(second) * stride;

      for (int b : adj_second) {
        if (b == start || b == second) {
          continue;
        }

        const float prefix2 = first_w + row_second[static_cast<std::size_t>(b)];
        const float* row_b = W.data() + static_cast<std::size_t>(b) * stride;
        const float32x4_t prefix_v = vdupq_n_f32(prefix2);
        const float32x4_t start_v = vdupq_n_f32(static_cast<float>(start));
        const float32x4_t second_v = vdupq_n_f32(static_cast<float>(second));
        const float32x4_t b_v = vdupq_n_f32(static_cast<float>(b));

        int j = 0;
        for (; j + 4 <= N; j += 4) {
          const float32x4_t idx_v =
              vld1q_f32(index_f.data() + static_cast<std::size_t>(j));

          uint32x4_t valid_v = vmvnq_u32(vceqq_f32(idx_v, start_v));
          valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, second_v)));
          valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, b_v)));

          const float32x4_t total_v = vaddq_f32(
              prefix_v,
              vaddq_f32(
                  vld1q_f32(row_b + static_cast<std::size_t>(j)),
                  vld1q_f32(close_to_start + static_cast<std::size_t>(j))));

          const int mask = neon_movemask_u32(vandq_u32(valid_v, vcltq_f32(total_v, zero_v)));

          if (mask != 0) {
            vst1q_f32(totals, total_v);
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
      const float* row_second =
          W.data() + static_cast<std::size_t>(second) * stride;

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

          const float32x4_t prefix_v = vdupq_n_f32(prefix3);
          const float32x4_t start_v = vdupq_n_f32(static_cast<float>(start));
          const float32x4_t second_v = vdupq_n_f32(static_cast<float>(second));
          const float32x4_t b_v = vdupq_n_f32(static_cast<float>(b));
          const float32x4_t c_v = vdupq_n_f32(static_cast<float>(c));

          int j = 0;
          for (; j + 4 <= N; j += 4) {
            const float32x4_t idx_v =
                vld1q_f32(index_f.data() + static_cast<std::size_t>(j));

            uint32x4_t valid_v = vmvnq_u32(vceqq_f32(idx_v, start_v));
            valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, second_v)));
            valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, b_v)));
            valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, c_v)));

            const float32x4_t total_v = vaddq_f32(
                prefix_v,
                vaddq_f32(
                    vld1q_f32(row_c + static_cast<std::size_t>(j)),
                    vld1q_f32(close_to_start + static_cast<std::size_t>(j))));

            const int mask = neon_movemask_u32(vandq_u32(valid_v, vcltq_f32(total_v, zero_v)));

            if (mask != 0) {
              vst1q_f32(totals, total_v);
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
        find_best_through_edge_neon(forward.from, forward.to));
  }

  if (reverse_improved) {
    result = better_optional(
        std::move(result),
        find_best_through_edge_neon(reverse.from, reverse.to));
  }

  cached_best_ = result;
  cached_max_cycle_length_ = max_cycle_length;
  cache_valid_ = true;
  return cached_best_;
}

std::optional<Cycle>
NeonArbitrageDetector::find_best_cycle_through_edge(
    int from,
    int to,
    int max_cycle_length) const {
  if (max_cycle_length < 2 || from < 0 || to < 0 || from >= n() || to >= n() ||
      from == to || !cell(from, to).exists) {
    return std::nullopt;
  }

  // Specialized NEON kernels for 5..10 only.
  // Keep exact scalar DFS for anything else.
  if (max_cycle_length < 5 || max_cycle_length > 10) {
    SearchState state(max_cycle_length);
    state.visited.assign(static_cast<std::size_t>(n()), 0);
    state.path.reserve(static_cast<std::size_t>(max_cycle_length) + 1);
    state.visited[static_cast<std::size_t>(from)] = 1;
    state.visited[static_cast<std::size_t>(to)] = 1;
    state.path.push_back(from);
    state.path.push_back(to);

    const QuoteCell& first = cell(from, to);
    dfs_from_fixed_edge(from, to, 1, first.weight, first.net_rate, state);
    return state.best;
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

  const float first_w =
      W[static_cast<std::size_t>(from) * stride + static_cast<std::size_t>(to)];
  if (!std::isfinite(first_w)) {
    return std::nullopt;
  }

  bool have_best = false;
  float best_weight = inf;
  int best_len = 0;
  int best_vertices[11] = {0};

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
      WT.data() + static_cast<std::size_t>(from) * stride;

  // Direct 2-cycle: from -> to -> from
  const float total2 = first_w + close_to_start[static_cast<std::size_t>(to)];
  if (total2 < 0.0f) {
    const int path[3] = {from, to, from};
    record_candidate(total2, 2, path);
  }

  const float32x4_t zero_v = vdupq_n_f32(0.0f);
  alignas(16) float totals[4];

  int prefix[10];
  prefix[0] = from;
  prefix[1] = to;

  auto scan_last_hop = [&](const int* pref,
                           int pref_size,
                           float prefix_weight,
                           const float* row_last) {
    // pref holds the vertices already in the path, including `from` and `to`,
    // and ending at the current last vertex. We scan one more vertex j, then
    // close j -> from. Total cycle length = pref_size + 1.
    const int cycle_len = pref_size + 1;

    float32x4_t banned_v[10];
    for (int i = 0; i < pref_size; ++i) {
      banned_v[i] = vdupq_n_f32(static_cast<float>(pref[i]));
    }

    const float32x4_t prefix_v = vdupq_n_f32(prefix_weight);

    int j = 0;
    for (; j + 4 <= N; j += 4) {
      const float32x4_t idx_v =
          vld1q_f32(index_f.data() + static_cast<std::size_t>(j));

      uint32x4_t valid_v = vmvnq_u32(vceqq_f32(idx_v, banned_v[0]));
      for (int i = 1; i < pref_size; ++i) {
        valid_v = vandq_u32(valid_v, vmvnq_u32(vceqq_f32(idx_v, banned_v[i])));
      }

      const float32x4_t total_v = vaddq_f32(
          prefix_v,
          vaddq_f32(
              vld1q_f32(row_last + static_cast<std::size_t>(j)),
              vld1q_f32(close_to_start + static_cast<std::size_t>(j))));

      const int mask = neon_movemask_u32(vandq_u32(valid_v, vcltq_f32(total_v, zero_v)));

      if (mask != 0) {
        vst1q_f32(totals, total_v);
        for (int lane = 0; lane < 4; ++lane) {
          if ((mask & (1 << lane)) == 0) {
            continue;
          }
          const int next = j + lane;
          int path[11];
          for (int i = 0; i < pref_size; ++i) {
            path[i] = pref[i];
          }
          path[pref_size] = next;
          path[pref_size + 1] = from;
          record_candidate(totals[lane], cycle_len, path);
        }
      }
    }

    for (; j < N; ++j) {
      bool banned = false;
      for (int i = 0; i < pref_size; ++i) {
        if (j == pref[i]) {
          banned = true;
          break;
        }
      }
      if (banned) {
        continue;
      }

      const float total =
          prefix_weight +
          row_last[static_cast<std::size_t>(j)] +
          close_to_start[static_cast<std::size_t>(j)];

      if (total < 0.0f) {
        int path[11];
        for (int i = 0; i < pref_size; ++i) {
          path[i] = pref[i];
        }
        path[pref_size] = j;
        path[pref_size + 1] = from;
        record_candidate(total, cycle_len, path);
      }
    }
  };

  const float* row_to = W.data() + static_cast<std::size_t>(to) * stride;

  // Length 3: from -> to -> b -> from
  if (max_cycle_length >= 3) {
    scan_last_hop(prefix, 2, first_w, row_to);
  }

  const auto& adj_to = outgoing_[static_cast<std::size_t>(to)];

  // Length 4: from -> to -> b -> c -> from
  if (max_cycle_length >= 4) {
    for (int b : adj_to) {
      if (b == from || b == to) {
        continue;
      }
      prefix[2] = b;
      const float prefix2 = first_w + row_to[static_cast<std::size_t>(b)];
      const float* row_b = W.data() + static_cast<std::size_t>(b) * stride;
      scan_last_hop(prefix, 3, prefix2, row_b);
    }
  }

  // Length 5+: keep unrolling outward, NEON-scanning the final hop.
  if (max_cycle_length >= 5) {
    for (int b : adj_to) {
      if (b == from || b == to) {
        continue;
      }
      prefix[2] = b;
      const float prefix2 = first_w + row_to[static_cast<std::size_t>(b)];
      const float* row_b = W.data() + static_cast<std::size_t>(b) * stride;
      const auto& adj_b = outgoing_[static_cast<std::size_t>(b)];

      // Length 5: from -> to -> b -> c -> d -> from
      for (int c : adj_b) {
        if (c == from || c == to || c == b) {
          continue;
        }
        prefix[3] = c;
        const float prefix3 =
            prefix2 + row_b[static_cast<std::size_t>(c)];
        const float* row_c = W.data() + static_cast<std::size_t>(c) * stride;
        scan_last_hop(prefix, 4, prefix3, row_c);

        if (max_cycle_length < 6) {
          continue;
        }

        const auto& adj_c = outgoing_[static_cast<std::size_t>(c)];

        // Length 6
        for (int d : adj_c) {
          if (d == from || d == to || d == b || d == c) {
            continue;
          }
          prefix[4] = d;
          const float prefix4 =
              prefix3 + row_c[static_cast<std::size_t>(d)];
          const float* row_d = W.data() + static_cast<std::size_t>(d) * stride;
          scan_last_hop(prefix, 5, prefix4, row_d);

          if (max_cycle_length < 7) {
            continue;
          }

          const auto& adj_d = outgoing_[static_cast<std::size_t>(d)];

          // Length 7
          for (int e : adj_d) {
            if (e == from || e == to || e == b || e == c || e == d) {
              continue;
            }
            prefix[5] = e;
            const float prefix5 =
                prefix4 + row_d[static_cast<std::size_t>(e)];
            const float* row_e = W.data() + static_cast<std::size_t>(e) * stride;
            scan_last_hop(prefix, 6, prefix5, row_e);

            if (max_cycle_length < 8) {
              continue;
            }

            const auto& adj_e = outgoing_[static_cast<std::size_t>(e)];

            // Length 8
            for (int f : adj_e) {
              if (f == from || f == to || f == b || f == c || f == d || f == e) {
                continue;
              }
              prefix[6] = f;
              const float prefix6 =
                  prefix5 + row_e[static_cast<std::size_t>(f)];
              const float* row_f = W.data() + static_cast<std::size_t>(f) * stride;
              scan_last_hop(prefix, 7, prefix6, row_f);

              if (max_cycle_length < 9) {
                continue;
              }

              const auto& adj_f = outgoing_[static_cast<std::size_t>(f)];

              // Length 9
              for (int g : adj_f) {
                if (g == from || g == to || g == b || g == c || g == d ||
                    g == e || g == f) {
                  continue;
                }
                prefix[7] = g;
                const float prefix7 =
                    prefix6 + row_f[static_cast<std::size_t>(g)];
                const float* row_g =
                    W.data() + static_cast<std::size_t>(g) * stride;
                scan_last_hop(prefix, 8, prefix7, row_g);

                if (max_cycle_length < 10) {
                  continue;
                }

                const auto& adj_g = outgoing_[static_cast<std::size_t>(g)];

                // Length 10
                for (int h : adj_g) {
                  if (h == from || h == to || h == b || h == c || h == d ||
                      h == e || h == f || h == g) {
                    continue;
                  }
                  prefix[8] = h;
                  const float prefix8 =
                      prefix7 + row_g[static_cast<std::size_t>(h)];
                  const float* row_h =
                      W.data() + static_cast<std::size_t>(h) * stride;
                  scan_last_hop(prefix, 9, prefix8, row_h);
                }
              }
            }
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


} // namespace arbcycle

namespace nb = nanobind;

NB_MODULE(_macos_arm64_neon, m) {
  arbcycle::bind_detector_module<arbcycle::NeonArbitrageDetector>(
      m,
      "_NeonArbitrageDetector",
      "Arm NEON bounded simple-cycle arbitrage detector");
}
