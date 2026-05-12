#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/types.h"

namespace negcycle {

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
  virtual ~ArbitrageDetectorBase() = default;

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

  [[nodiscard]] virtual std::optional<Cycle> find_best_arbitrage(int max_cycle_length);
  [[nodiscard]] virtual std::optional<Cycle> add_quote_and_find_best_arbitrage(
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length);
  [[nodiscard]] virtual std::optional<Cycle> add_book_and_find_best_arbitrage(
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

protected:
  static constexpr float kCompareEpsilon = 1.0e-7f;

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

  [[nodiscard]] int n() const noexcept { return static_cast<int>(codes_.size()); }
  [[nodiscard]] const QuoteCell& cell(int from, int to) const noexcept;
  [[nodiscard]] QuoteCell& cell(int from, int to) noexcept;

  void resize_storage(int new_n);
  void invalidate_cache() noexcept;
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
  [[nodiscard]] virtual std::optional<Cycle> find_best_cycle_through_edge(
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

  bool cache_valid_{false};
  int cached_max_cycle_length_{-1};
  std::optional<Cycle> cached_best_;
};

} // namespace negcycle
