#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/types.h"

namespace negcycle {

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n == 0) {
      return nullptr;
    }
    const std::size_t raw = n * sizeof(T);
    const std::size_t bytes = (raw + Alignment - 1) & ~(Alignment - 1);
    void* p = std::aligned_alloc(Alignment, bytes);
    if (!p) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(p);
  }

  void deallocate(T* p, std::size_t) noexcept {
    std::free(p);
  }

  template <typename U>
  struct rebind { using other = AlignedAllocator<U, Alignment>; };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&,
                const AlignedAllocator<U, Alignment>&) noexcept { return true; }
template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>& lhs,
                const AlignedAllocator<U, Alignment>& rhs) noexcept {
  return !(lhs == rhs);
}

template <typename Detector, typename Traits, typename ScanPolicy>
struct SimdSearch;

class ArbitrageDetectorBase {
public:
  using Edge = ::negcycle::Edge;
  using Cycle = ::negcycle::Cycle;

  struct SerializedQuote {
    std::string from;
    std::string to;
    float gross_rate{0.0f};
    float fee_bps{0.0f};
  };

  ArbitrageDetectorBase() = default;
  ArbitrageDetectorBase(const ArbitrageDetectorBase&) = default;
  ArbitrageDetectorBase(ArbitrageDetectorBase&&) noexcept = default;
  ArbitrageDetectorBase& operator=(const ArbitrageDetectorBase&) = default;
  ArbitrageDetectorBase& operator=(ArbitrageDetectorBase&&) noexcept = default;
  ~ArbitrageDetectorBase() = default;

  [[nodiscard]] int add_currency(std::string_view code);
  void add_quote(std::string_view from,
                 std::string_view to,
                 float executable_rate,
                 float fee_bps = 0.0f);
  void add_book(std::string_view base,
                std::string_view quote,
                float bid,
                float ask,
                float fee_bps = 0.0f);

  [[nodiscard]] std::optional<Cycle> find_best_arbitrage(int max_cycle_length);
  [[nodiscard]] std::optional<Cycle> add_quote_and_find_best_arbitrage(
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length);
  [[nodiscard]] std::optional<Cycle> add_book_and_find_best_arbitrage(
      std::string_view base,
      std::string_view quote,
      float bid,
      float ask,
      float fee_bps,
      int max_cycle_length);

  [[nodiscard]] std::vector<Cycle> find_arbitrage(int max_cycle_length);
  [[nodiscard]] std::vector<Cycle> add_quote_and_find_arbitrage(
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length);
  [[nodiscard]] std::vector<Cycle> add_book_and_find_arbitrage(
      std::string_view base,
      std::string_view quote,
      float bid,
      float ask,
      float fee_bps,
      int max_cycle_length);

  [[nodiscard]] const std::vector<std::string>& currencies() const noexcept {
    return codes_;
  }
  [[nodiscard]] bool has_currency(std::string_view code) const noexcept;
  [[nodiscard]] bool has_quote(std::string_view from, std::string_view to) const noexcept;
  [[nodiscard]] std::vector<SerializedQuote> serialized_quotes() const;
  void restore_state(const std::vector<std::string>& currencies,
                     const std::vector<SerializedQuote>& quotes);

  // Public so scan policies (defined outside this class) can reuse the same
  // comparison tolerance when deriving their pruning threshold.
  static constexpr float kCompareEpsilon = 1.0e-7f;

protected:

  struct QuoteCell {
    bool exists{false};
    int from{-1};
    int to{-1};
    float gross_rate{0.0f};
    float fee_bps{0.0f};
    float net_rate{0.0f};
    float weight{0.0f};
  };

  struct UpsertResult {
    int from{-1};
    int to{-1};
    bool new_currency{false};
    bool existed{false};
    bool new_edge{false};
    float old_weight{0.0f};
    float new_weight{0.0f};
  };

  struct SearchState {
    int max_cycle_length{0};
    std::vector<unsigned char> visited;
    std::vector<int> path;
    std::optional<Cycle> best;

    explicit SearchState(int max_cycle_length_) : max_cycle_length(max_cycle_length_) {}
  };

  struct AllCyclesState {
    int max_cycle_length{0};
    std::vector<unsigned char> visited;
    std::vector<int> path;
    std::vector<Cycle> cycles;

    explicit AllCyclesState(int max_cycle_length_) : max_cycle_length(max_cycle_length_) {}
  };

  struct DenseWeights {
    // 64 satisfies AVX-512 alignment AND keeps every row 64-byte aligned
    // whenever `padded_stride * sizeof(float)` is a multiple of 64. For
    // detectors that don't request padding (`dense_pad_multiple() == 1`)
    // `padded_stride == n` and only the buffer start is guaranteed aligned.
    static constexpr std::size_t kAlignment = 64;

    using Buffer = std::vector<float, AlignedAllocator<float, kAlignment>>;

    int n{0};
    int padded_stride{0};
    Buffer weights;
    Buffer transpose;
    bool valid{false};
  };

  [[nodiscard]] int n() const noexcept { return static_cast<int>(codes_.size()); }
  [[nodiscard]] const QuoteCell& cell(int from, int to) const noexcept;
  [[nodiscard]] QuoteCell& cell(int from, int to) noexcept;
  // `pad_multiple` rounds each row up to a multiple of this many floats and
  // fills the trailing cells with +inf. The SIMD inner loop can then run
  // without a scalar tail. Default 1 means no padding (rows are `n` long).
  [[nodiscard]] const DenseWeights& dense_weights(int pad_multiple = 1) const;

  void resize_storage(int new_n);
  void invalidate_cache() noexcept;
  void invalidate_dense_weights() noexcept;
  [[nodiscard]] UpsertResult upsert_quote(std::string_view from,
                                          std::string_view to,
                                          float executable_rate,
                                          float fee_bps);

  void dfs_from_start(int start,
                      int current,
                      int depth_used,
                      float path_weight,
                      float path_gain,
                      SearchState& state) const;
  void dfs_from_fixed_edge(int start,
                           int current,
                           int depth_used,
                           float path_weight,
                           float path_gain,
                           SearchState& state) const;
  void dfs_all_from_start(int start,
                          int current,
                          int depth_used,
                          float path_weight,
                          float path_gain,
                          AllCyclesState& state) const;

  [[nodiscard]] std::optional<Cycle> find_best_arbitrage_common(int max_cycle_length);
  [[nodiscard]] std::optional<Cycle> add_quote_and_find_best_arbitrage_common(
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length);
  [[nodiscard]] std::optional<Cycle> add_book_and_find_best_arbitrage_common(
      std::string_view base,
      std::string_view quote,
      float bid,
      float ask,
      float fee_bps,
      int max_cycle_length);
  [[nodiscard]] std::vector<Cycle> find_arbitrage_common(int max_cycle_length);
  [[nodiscard]] std::vector<Cycle> add_quote_and_find_arbitrage_common(
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length);
  [[nodiscard]] std::vector<Cycle> add_book_and_find_arbitrage_common(
      std::string_view base,
      std::string_view quote,
      float bid,
      float ask,
      float fee_bps,
      int max_cycle_length);

  [[nodiscard]] std::optional<Cycle> find_best_cycle_through_edge_scalar(
      int from,
      int to,
      int max_cycle_length) const;
  [[nodiscard]] std::optional<Cycle> find_best_cycle_through_edge(
      int from,
      int to,
      int max_cycle_length) const;

  [[nodiscard]] Cycle materialize_cycle(const std::vector<int>& path,
                                        float total_weight,
                                        float gain_factor) const;
  [[nodiscard]] bool cycle_uses_edge(const Cycle& cycle, int from, int to) const noexcept;
  [[nodiscard]] static bool is_better_cycle(const Cycle& lhs, const Cycle& rhs) noexcept;
  [[nodiscard]] static std::optional<Cycle> better_optional(std::optional<Cycle> lhs,
                                                            std::optional<Cycle> rhs) noexcept;

  std::vector<std::string> codes_;
  std::unordered_map<std::string, int> id_by_code_;
  std::vector<QuoteCell> cells_;
  std::vector<std::vector<int>> outgoing_;
  mutable DenseWeights dense_weights_;

  bool cache_valid_{false};
  int cached_max_cycle_length_{-1};
  std::optional<Cycle> cached_best_;

  template <typename Detector, typename Traits, typename ScanPolicy>
  friend struct SimdSearch;
};

} // namespace negcycle
