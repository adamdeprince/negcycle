#include <optional>
#include <string_view>
#include <vector>

#include "backends/x86_simd_search.hpp"
#include "common/detector_bindings.h"

namespace negcycle {

class LinuxAarch64SveArbitrageDetector final : public ArbitrageDetectorBase {
public:
  using Search = SimdSearch<LinuxAarch64SveArbitrageDetector, PortableSimd256Traits>;

  [[nodiscard]] std::optional<Cycle> find_best_arbitrage(int max_cycle_length) {
    return Search::find_best_arbitrage(*this, max_cycle_length);
  }

  [[nodiscard]] std::optional<Cycle> add_quote_and_find_best_arbitrage(
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length) {
    return Search::add_quote_and_find_best_arbitrage(
        *this, from, to, executable_rate, fee_bps, max_cycle_length);
  }

  [[nodiscard]] std::optional<Cycle> add_book_and_find_best_arbitrage(
      std::string_view base,
      std::string_view quote,
      float bid,
      float ask,
      float fee_bps,
      int max_cycle_length) {
    return Search::add_book_and_find_best_arbitrage(
        *this, base, quote, bid, ask, fee_bps, max_cycle_length);
  }

  [[nodiscard]] std::vector<Cycle> find_arbitrage(int max_cycle_length) {
    return Search::find_arbitrage(*this, max_cycle_length);
  }

  [[nodiscard]] std::vector<Cycle> add_quote_and_find_arbitrage(
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length) {
    return Search::add_quote_and_find_arbitrage(
        *this, from, to, executable_rate, fee_bps, max_cycle_length);
  }

  [[nodiscard]] std::vector<Cycle> add_book_and_find_arbitrage(
      std::string_view base,
      std::string_view quote,
      float bid,
      float ask,
      float fee_bps,
      int max_cycle_length) {
    return Search::add_book_and_find_arbitrage(
        *this, base, quote, bid, ask, fee_bps, max_cycle_length);
  }
};

} // namespace negcycle

namespace nb = nanobind;

NB_MODULE(_linux_aarch64_sve, m) {
  negcycle::bind_detector_module<negcycle::LinuxAarch64SveArbitrageDetector>(
      m,
      "_LinuxAarch64SveArbitrageDetector",
      "Linux AArch64 SVE bounded simple-cycle arbitrage detector");
}
