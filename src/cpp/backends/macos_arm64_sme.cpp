#include "common/detector_base.h"
#include "common/detector_bindings.h"

namespace arbcycle {

class SmeArbitrageDetector final : public ArbitrageDetectorBase {
public:
  [[nodiscard]] std::optional<Cycle> find_best_arbitrage(int max_cycle_length) override {
    return find_best_arbitrage_common(max_cycle_length);
  }

  [[nodiscard]] std::optional<Cycle> add_quote_and_find_best_arbitrage(
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length) override {
    return add_quote_and_find_best_arbitrage_common(
        from, to, executable_rate, fee_bps, max_cycle_length);
  }

  [[nodiscard]] std::optional<Cycle> add_book_and_find_best_arbitrage(
      std::string_view base,
      std::string_view quote,
      float bid,
      float ask,
      float fee_bps,
      int max_cycle_length) override {
    return add_book_and_find_best_arbitrage_common(
        base, quote, bid, ask, fee_bps, max_cycle_length);
  }
};

} // namespace arbcycle

namespace nb = nanobind;

NB_MODULE(_macos_arm64_sme, m) {
  arbcycle::bind_detector_module<arbcycle::SmeArbitrageDetector>(
      m,
      "_SmeArbitrageDetector",
      "Arm SME bounded simple-cycle arbitrage detector");
}
