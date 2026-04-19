#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/detector_base.h"
#include "common/detector_bindings.h"

namespace arbcycle {

class AvxArbitrageDetector final : public ArbitrageDetectorBase {
protected:


} // namespace arbcycle

namespace nb = nanobind;

NB_MODULE(_avx, m) {
  arbcycle::bind_detector_module<arbcycle::AvxArbitrageDetector>(
      m,
      "_AvxArbitrageDetector",
      "AVX-accelerated bounded simple-cycle arbitrage detector");
}
