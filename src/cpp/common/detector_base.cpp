#include "common/detector_base.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace negcycle {
namespace {

[[nodiscard]] bool lexicographically_smaller(const std::vector<int>& a,
                                             const std::vector<int>& b) noexcept {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

} // namespace

int ArbitrageDetectorBase::add_currency(std::string_view code) {
  const std::string key(code);
  if (const auto it = id_by_code_.find(key); it != id_by_code_.end()) {
    return it->second;
  }

  const int id = n();
  codes_.push_back(key);
  id_by_code_.emplace(codes_.back(), id);
  resize_storage(id + 1);
  invalidate_cache();
  return id;
}

void ArbitrageDetectorBase::add_quote(std::string_view from,
                                      std::string_view to,
                                      float executable_rate,
                                      float fee_bps) {
  (void) upsert_quote(from, to, executable_rate, fee_bps);
  invalidate_cache();
}

void ArbitrageDetectorBase::add_book(std::string_view base,
                                     std::string_view quote,
                                     float bid,
                                     float ask,
                                     float fee_bps) {
  if (!(bid > 0.0f) || !(ask > 0.0f) || bid > ask) {
    throw std::invalid_argument("invalid bid/ask");
  }
  add_quote(base, quote, bid, fee_bps);
  add_quote(quote, base, 1.0f / ask, fee_bps);
}

bool ArbitrageDetectorBase::has_currency(std::string_view code) const noexcept {
  return id_by_code_.find(std::string(code)) != id_by_code_.end();
}

bool ArbitrageDetectorBase::has_quote(std::string_view from, std::string_view to) const noexcept {
  const auto it_from = id_by_code_.find(std::string(from));
  if (it_from == id_by_code_.end()) {
    return false;
  }

  const auto it_to = id_by_code_.find(std::string(to));
  if (it_to == id_by_code_.end()) {
    return false;
  }

  return cell(it_from->second, it_to->second).exists;
}

std::vector<ArbitrageDetectorBase::SerializedQuote> ArbitrageDetectorBase::serialized_quotes() const {
  std::vector<SerializedQuote> out;
  out.reserve(cells_.size());

  for (int i = 0; i < n(); ++i) {
    for (int j = 0; j < n(); ++j) {
      const QuoteCell& q = cell(i, j);
      if (!q.exists) {
        continue;
      }

      out.push_back(SerializedQuote{
          .from = codes_[static_cast<std::size_t>(q.from)],
          .to = codes_[static_cast<std::size_t>(q.to)],
          .gross_rate = q.gross_rate,
          .fee_bps = q.fee_bps,
      });
    }
  }

  return out;
}

void ArbitrageDetectorBase::restore_state(const std::vector<std::string>& currencies,
                                          const std::vector<SerializedQuote>& quotes) {
  codes_.clear();
  id_by_code_.clear();
  cells_.clear();
  outgoing_.clear();
  invalidate_dense_weights();
  invalidate_cache();

  for (const std::string& code : currencies) {
    (void) add_currency(code);
  }

  for (const SerializedQuote& quote : quotes) {
    (void) upsert_quote(quote.from, quote.to, quote.gross_rate, quote.fee_bps);
  }

  invalidate_cache();
}

std::optional<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::find_best_arbitrage(int max_cycle_length) {
  return find_best_arbitrage_common(max_cycle_length);
}

std::optional<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::add_quote_and_find_best_arbitrage(std::string_view from,
                                                         std::string_view to,
                                                         float executable_rate,
                                                         float fee_bps,
                                                         int max_cycle_length) {
  return add_quote_and_find_best_arbitrage_common(
      from, to, executable_rate, fee_bps, max_cycle_length);
}

std::optional<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::add_book_and_find_best_arbitrage(std::string_view base,
                                                        std::string_view quote,
                                                        float bid,
                                                        float ask,
                                                        float fee_bps,
                                                        int max_cycle_length) {
  return add_book_and_find_best_arbitrage_common(
      base, quote, bid, ask, fee_bps, max_cycle_length);
}

std::vector<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::find_arbitrage(int max_cycle_length) {
  return find_arbitrage_common(max_cycle_length);
}

std::vector<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::add_quote_and_find_arbitrage(std::string_view from,
                                                    std::string_view to,
                                                    float executable_rate,
                                                    float fee_bps,
                                                    int max_cycle_length) {
  return add_quote_and_find_arbitrage_common(
      from, to, executable_rate, fee_bps, max_cycle_length);
}

std::vector<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::add_book_and_find_arbitrage(std::string_view base,
                                                   std::string_view quote,
                                                   float bid,
                                                   float ask,
                                                   float fee_bps,
                                                   int max_cycle_length) {
  return add_book_and_find_arbitrage_common(
      base, quote, bid, ask, fee_bps, max_cycle_length);
}

std::optional<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::find_best_arbitrage_common(int max_cycle_length) {
  if (max_cycle_length < 2 || n() < 2) {
    cached_best_.reset();
    cached_max_cycle_length_ = max_cycle_length;
    cache_valid_ = true;
    return std::nullopt;
  }

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

std::vector<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::find_arbitrage_common(int max_cycle_length) {
  if (max_cycle_length < 2 || n() < 2) {
    return {};
  }

  AllCyclesState state(max_cycle_length);
  state.visited.assign(static_cast<std::size_t>(n()), 0);
  state.path.reserve(static_cast<std::size_t>(max_cycle_length) + 1);

  for (int start = 0; start < n(); ++start) {
    std::fill(state.visited.begin(), state.visited.end(), static_cast<unsigned char>(0));
    state.path.clear();
    state.path.push_back(start);
    state.visited[static_cast<std::size_t>(start)] = 1;
    dfs_all_from_start(start, start, 0, 0.0f, 1.0f, state);
  }

  std::sort(state.cycles.begin(), state.cycles.end(), is_better_cycle);
  return state.cycles;
}

std::vector<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::add_quote_and_find_arbitrage_common(std::string_view from,
                                                           std::string_view to,
                                                           float executable_rate,
                                                           float fee_bps,
                                                           int max_cycle_length) {
  (void) upsert_quote(from, to, executable_rate, fee_bps);
  invalidate_cache();
  return find_arbitrage_common(max_cycle_length);
}

std::vector<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::add_book_and_find_arbitrage_common(std::string_view base,
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

  (void) upsert_quote(base, quote, bid, fee_bps);
  (void) upsert_quote(quote, base, 1.0f / ask, fee_bps);
  invalidate_cache();
  return find_arbitrage_common(max_cycle_length);
}

std::optional<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::add_quote_and_find_best_arbitrage_common(std::string_view from,
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

  const bool improved = update.new_edge || update.new_weight < update.old_weight - kCompareEpsilon;
  const bool worsened = update.existed && update.new_weight > update.old_weight + kCompareEpsilon;
  const bool cached_uses_edge =
      cached_best_.has_value() && cycle_uses_edge(*cached_best_, update.from, update.to);

  if (improved) {
    std::optional<Cycle> through_edge =
        find_best_cycle_through_edge(update.from, update.to, max_cycle_length);
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

std::optional<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::add_book_and_find_best_arbitrage_common(std::string_view base,
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
    return add_quote_and_find_best_arbitrage(
        quote, base, reverse_rate, fee_bps, max_cycle_length);
  }

  if (!forward_same && reverse_same) {
    return add_quote_and_find_best_arbitrage(base, quote, bid, fee_bps, max_cycle_length);
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
      cached_best_.has_value() && cycle_uses_edge(*cached_best_, forward.from, forward.to);
  const bool cached_uses_reverse =
      cached_best_.has_value() && cycle_uses_edge(*cached_best_, reverse.from, reverse.to);

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

const ArbitrageDetectorBase::QuoteCell& ArbitrageDetectorBase::cell(int from, int to) const noexcept {
  return cells_[static_cast<std::size_t>(from) * static_cast<std::size_t>(n()) +
                static_cast<std::size_t>(to)];
}

ArbitrageDetectorBase::QuoteCell& ArbitrageDetectorBase::cell(int from, int to) noexcept {
  return cells_[static_cast<std::size_t>(from) * static_cast<std::size_t>(n()) +
                static_cast<std::size_t>(to)];
}

void ArbitrageDetectorBase::resize_storage(int new_n) {
  const int old_n = static_cast<int>(outgoing_.size());
  std::vector<QuoteCell> new_cells(static_cast<std::size_t>(new_n) *
                                   static_cast<std::size_t>(new_n));

  for (int i = 0; i < new_n; ++i) {
    for (int j = 0; j < new_n; ++j) {
      QuoteCell& dst =
          new_cells[static_cast<std::size_t>(i) * static_cast<std::size_t>(new_n) +
                    static_cast<std::size_t>(j)];
      dst.exists = false;
      dst.from = i;
      dst.to = j;
      dst.weight = std::numeric_limits<float>::infinity();
    }
  }

  for (int i = 0; i < old_n; ++i) {
    for (int j = 0; j < old_n; ++j) {
      new_cells[static_cast<std::size_t>(i) * static_cast<std::size_t>(new_n) +
                static_cast<std::size_t>(j)] =
          cells_[static_cast<std::size_t>(i) * static_cast<std::size_t>(old_n) +
                 static_cast<std::size_t>(j)];
    }
  }

  cells_.swap(new_cells);
  outgoing_.resize(static_cast<std::size_t>(new_n));
  invalidate_dense_weights();
}

void ArbitrageDetectorBase::invalidate_cache() noexcept {
  cache_valid_ = false;
  cached_max_cycle_length_ = -1;
  cached_best_.reset();
}

void ArbitrageDetectorBase::invalidate_dense_weights() noexcept {
  dense_weights_.valid = false;
  dense_weights_.n = 0;
  dense_weights_.padded_stride = 0;
}

ArbitrageDetectorBase::UpsertResult ArbitrageDetectorBase::upsert_quote(std::string_view from,
                                                                        std::string_view to,
                                                                        float executable_rate,
                                                                        float fee_bps) {
  if (!(executable_rate > 0.0f)) {
    throw std::invalid_argument("rate must be > 0");
  }
  if (!(fee_bps >= 0.0f) || fee_bps >= 10000.0f) {
    throw std::invalid_argument("fee_bps must be in [0, 10000)");
  }

  const int old_n = n();
  const int from_id = add_currency(from);
  const int to_id = add_currency(to);

  QuoteCell& q = cell(from_id, to_id);
  const bool existed = q.exists;
  const float old_weight = q.weight;

  const float fee_frac = fee_bps * 1.0e-4f;
  const float net_rate = executable_rate * (1.0f - fee_frac);
  if (!(net_rate > 0.0f)) {
    throw std::invalid_argument("effective rate must be > 0");
  }

  if (!existed) {
    auto& adj = outgoing_[static_cast<std::size_t>(from_id)];
    if (std::find(adj.begin(), adj.end(), to_id) == adj.end()) {
      adj.push_back(to_id);
      std::sort(adj.begin(), adj.end());
    }
  }

  q.exists = true;
  q.from = from_id;
  q.to = to_id;
  q.gross_rate = executable_rate;
  q.fee_bps = fee_bps;
  q.net_rate = net_rate;
  q.weight = -static_cast<float>(std::log(static_cast<double>(net_rate)));

  if (dense_weights_.valid && dense_weights_.n == n()) {
    const std::size_t stride = static_cast<std::size_t>(dense_weights_.padded_stride);
    const std::size_t from_index = static_cast<std::size_t>(from_id);
    const std::size_t to_index = static_cast<std::size_t>(to_id);
    dense_weights_.weights[from_index * stride + to_index] = q.weight;
    dense_weights_.transpose[to_index * stride + from_index] = q.weight;
  }

  return UpsertResult{
      .from = from_id,
      .to = to_id,
      .new_currency = n() != old_n,
      .existed = existed,
      .new_edge = !existed,
      .old_weight = old_weight,
      .new_weight = q.weight,
  };
}

const ArbitrageDetectorBase::DenseWeights& ArbitrageDetectorBase::dense_weights(int pad_multiple) const {
  const int N = n();
  const int K = pad_multiple > 0 ? pad_multiple : 1;
  // For K > 1, the padded SIMD kernel runs `for (j; j + K <= P; j += K)` with
  // no scalar tail, so the last iteration starts at P - K. To cover every
  // candidate j in [0, N) we need P - K >= N - 1, i.e. P >= N + K - 1.
  // Round up to the next multiple of K so loads stay block-aligned.
  const int P = K > 1 ? ((N + 2 * K - 2) / K) * K : N;
  if (dense_weights_.valid && dense_weights_.n == N && dense_weights_.padded_stride == P) {
    return dense_weights_;
  }
  const std::size_t row_stride = static_cast<std::size_t>(P);
  const std::size_t total = static_cast<std::size_t>(N) * row_stride;
  const float inf = std::numeric_limits<float>::infinity();

  dense_weights_.n = N;
  dense_weights_.padded_stride = P;
  dense_weights_.weights.assign(total, inf);
  dense_weights_.transpose.assign(total, inf);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      const QuoteCell& q = cell(i, j);
      if (!q.exists) {
        continue;
      }
      const std::size_t ij =
          static_cast<std::size_t>(i) * row_stride + static_cast<std::size_t>(j);
      const std::size_t ji =
          static_cast<std::size_t>(j) * row_stride + static_cast<std::size_t>(i);
      dense_weights_.weights[ij] = q.weight;
      dense_weights_.transpose[ji] = q.weight;
    }
  }

  dense_weights_.valid = true;
  return dense_weights_;
}

void ArbitrageDetectorBase::dfs_from_start(int start,
                                           int current,
                                           int depth_used,
                                           float path_weight,
                                           float path_gain,
                                           SearchState& state) const {
  if (depth_used >= 1) {
    const QuoteCell& close = cell(current, start);
    if (close.exists) {
      const float total_weight = path_weight + close.weight;
      if (total_weight < 0.0f) {
        const float gain_factor = path_gain * close.net_rate;
        std::vector<int> closed_path = state.path;
        closed_path.push_back(start);
        Cycle candidate = materialize_cycle(closed_path, total_weight, gain_factor);
        if (!state.best || is_better_cycle(candidate, *state.best)) {
          state.best = std::move(candidate);
        }
      }
    }
  }

  if (depth_used == state.max_cycle_length - 1) {
    return;
  }

  for (int next : outgoing_[static_cast<std::size_t>(current)]) {
    if (next <= start) {
      continue; // canonicalization: start must be the minimum vertex id in the cycle
    }
    if (state.visited[static_cast<std::size_t>(next)] != 0) {
      continue;
    }

    const QuoteCell& e = cell(current, next);
    state.visited[static_cast<std::size_t>(next)] = 1;
    state.path.push_back(next);
    dfs_from_start(start,
                   next,
                   depth_used + 1,
                   path_weight + e.weight,
                   path_gain * e.net_rate,
                   state);
    state.path.pop_back();
    state.visited[static_cast<std::size_t>(next)] = 0;
  }
}

void ArbitrageDetectorBase::dfs_all_from_start(int start,
                                               int current,
                                               int depth_used,
                                               float path_weight,
                                               float path_gain,
                                               AllCyclesState& state) const {
  if (depth_used >= 1) {
    const QuoteCell& close = cell(current, start);
    if (close.exists) {
      const float total_weight = path_weight + close.weight;
      if (total_weight < 0.0f) {
        const float gain_factor = path_gain * close.net_rate;
        std::vector<int> closed_path = state.path;
        closed_path.push_back(start);
        state.cycles.push_back(materialize_cycle(closed_path, total_weight, gain_factor));
      }
    }
  }

  if (depth_used == state.max_cycle_length - 1) {
    return;
  }

  for (int next : outgoing_[static_cast<std::size_t>(current)]) {
    if (next <= start) {
      continue; // canonicalization: start must be the minimum vertex id in the cycle
    }
    if (state.visited[static_cast<std::size_t>(next)] != 0) {
      continue;
    }

    const QuoteCell& e = cell(current, next);
    state.visited[static_cast<std::size_t>(next)] = 1;
    state.path.push_back(next);
    dfs_all_from_start(start,
                       next,
                       depth_used + 1,
                       path_weight + e.weight,
                       path_gain * e.net_rate,
                       state);
    state.path.pop_back();
    state.visited[static_cast<std::size_t>(next)] = 0;
  }
}

void ArbitrageDetectorBase::dfs_from_fixed_edge(int start,
                                                int current,
                                                int depth_used,
                                                float path_weight,
                                                float path_gain,
                                                SearchState& state) const {
  const QuoteCell& close = cell(current, start);
  if (close.exists) {
    const float total_weight = path_weight + close.weight;
    if (total_weight < 0.0f) {
      const float gain_factor = path_gain * close.net_rate;
      std::vector<int> closed_path = state.path;
      closed_path.push_back(start);
      Cycle candidate = materialize_cycle(closed_path, total_weight, gain_factor);
      if (!state.best || is_better_cycle(candidate, *state.best)) {
        state.best = std::move(candidate);
      }
    }
  }

  if (depth_used == state.max_cycle_length - 1) {
    return;
  }

  for (int next : outgoing_[static_cast<std::size_t>(current)]) {
    if (next == start) {
      continue; // handled explicitly above as cycle closure
    }
    if (state.visited[static_cast<std::size_t>(next)] != 0) {
      continue;
    }

    const QuoteCell& e = cell(current, next);
    state.visited[static_cast<std::size_t>(next)] = 1;
    state.path.push_back(next);
    dfs_from_fixed_edge(start,
                        next,
                        depth_used + 1,
                        path_weight + e.weight,
                        path_gain * e.net_rate,
                        state);
    state.path.pop_back();
    state.visited[static_cast<std::size_t>(next)] = 0;
  }
}

std::optional<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::find_best_cycle_through_edge_scalar(int from,
                                                           int to,
                                                           int max_cycle_length) const {
  if (max_cycle_length < 2 || !cell(from, to).exists) {
    return std::nullopt;
  }

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

std::optional<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::find_best_cycle_through_edge(int from,
                                                    int to,
                                                    int max_cycle_length) const {
  return find_best_cycle_through_edge_scalar(from, to, max_cycle_length);
}

ArbitrageDetectorBase::Cycle ArbitrageDetectorBase::materialize_cycle(
    const std::vector<int>& path,
    float total_weight,
    float gain_factor) const {
  Cycle out;
  out.vertices = path;
  out.names.reserve(path.size());
  out.legs.reserve(path.size() > 0 ? path.size() - 1 : 0);

  for (int vertex : path) {
    out.names.push_back(codes_[static_cast<std::size_t>(vertex)]);
  }

  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    const QuoteCell& q = cell(path[i], path[i + 1]);
    out.legs.push_back(Edge{
        .from = codes_[static_cast<std::size_t>(q.from)],
        .to = codes_[static_cast<std::size_t>(q.to)],
        .gross_rate = q.gross_rate,
        .fee_bps = q.fee_bps,
        .net_rate = q.net_rate,
        .weight = q.weight,
    });
  }

  out.total_weight = total_weight;
  out.log_gain = -total_weight;
  out.gain_factor = gain_factor;
  out.pct_return = (gain_factor - 1.0f) * 100.0f;
  return out;
}

bool ArbitrageDetectorBase::cycle_uses_edge(const Cycle& cycle, int from, int to) const noexcept {
  for (std::size_t i = 0; i + 1 < cycle.vertices.size(); ++i) {
    if (cycle.vertices[i] == from && cycle.vertices[i + 1] == to) {
      return true;
    }
  }
  return false;
}

bool ArbitrageDetectorBase::is_better_cycle(const Cycle& lhs, const Cycle& rhs) noexcept {
  if (lhs.total_weight < rhs.total_weight - kCompareEpsilon) {
    return true;
  }
  if (rhs.total_weight < lhs.total_weight - kCompareEpsilon) {
    return false;
  }

  const int lhs_len = lhs.length();
  const int rhs_len = rhs.length();
  if (lhs_len != rhs_len) {
    return lhs_len < rhs_len;
  }

  return lexicographically_smaller(lhs.vertices, rhs.vertices);
}

std::optional<ArbitrageDetectorBase::Cycle>
ArbitrageDetectorBase::better_optional(std::optional<Cycle> lhs,
                                       std::optional<Cycle> rhs) noexcept {
  if (!lhs) {
    return rhs;
  }
  if (!rhs) {
    return lhs;
  }
  return is_better_cycle(*lhs, *rhs) ? lhs : rhs;
}

} // namespace negcycle
