#include "common/detector_base.h"
#include "common/detector_bindings.h"

namespace arbcycle {

class GenericArbitrageDetector final : public ArbitrageDetectorBase {
};

} // namespace arbcycle

namespace nb = nanobind;

NB_MODULE(_generic, m) {
  arbcycle::bind_detector_module<arbcycle::GenericArbitrageDetector>(
      m,
      "_GenericArbitrageDetector",
      "Scalar bounded simple-cycle arbitrage detector");
}
