#include <optional>
#include <string_view>
#include <vector>

#include "backends/x86_simd_search.hpp"
#include "common/detector_bindings.h"

namespace negcycle {

class Avx512ArbitrageDetector final : public ArbitrageDetectorBase {
public:
  using Search = X86SimdSearch<Avx512ArbitrageDetector, Avx512SimdTraits>;

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

NB_MODULE(_avx512, m) {
  negcycle::bind_detector_module<negcycle::Avx512ArbitrageDetector>(
      m,
      "_Avx512ArbitrageDetector",
      "Experimental AVX512 bounded simple-cycle arbitrage detector");
}
