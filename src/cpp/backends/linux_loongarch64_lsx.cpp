#include "common/detector_base.h"
#include "common/detector_bindings.h"

namespace negcycle {

class LsxArbitrageDetector final : public ArbitrageDetectorBase {
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

  [[nodiscard]] std::vector<Cycle> find_arbitrage(int max_cycle_length) {
    return find_arbitrage_common(max_cycle_length);
  }

  [[nodiscard]] std::vector<Cycle> add_quote_and_find_arbitrage(
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length) {
    return add_quote_and_find_arbitrage_common(
        from, to, executable_rate, fee_bps, max_cycle_length);
  }

  [[nodiscard]] std::vector<Cycle> add_book_and_find_arbitrage(
      std::string_view base,
      std::string_view quote,
      float bid,
      float ask,
      float fee_bps,
      int max_cycle_length) {
    return add_book_and_find_arbitrage_common(
        base, quote, bid, ask, fee_bps, max_cycle_length);
  }
};

} // namespace negcycle

namespace nb = nanobind;

NB_MODULE(_linux_loongarch64_lsx, m) {
  negcycle::bind_detector_module<negcycle::LsxArbitrageDetector>(
      m,
      "_LsxArbitrageDetector",
      "LoongArch LSX bounded simple-cycle arbitrage detector");
}
