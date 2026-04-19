#pragma once

#include <string>
#include <vector>

namespace arbcycle {

struct Edge {
  int from{-1};
  int to{-1};
  float gross_rate{0.0f};
  float fee_bps{0.0f};
  float net_rate{0.0f};
  float weight{0.0f}; // -log(net_rate)
};

struct Cycle {
  std::vector<int> vertices;       // closed path: v0, v1, ..., v0
  std::vector<std::string> names;  // currency codes parallel to vertices
  std::vector<Edge> legs;          // one per edge in the cycle
  float total_weight{0.0f};        // sum of -log(net_rate), arbitrage iff < 0
  float log_gain{0.0f};            // -total_weight
  float gain_factor{1.0f};         // product of net rates
  float pct_return{0.0f};          // (gain_factor - 1) * 100

  [[nodiscard]] int length() const noexcept {
    return vertices.empty() ? 0 : static_cast<int>(vertices.size()) - 1;
  }
};

} // namespace arbcycle
