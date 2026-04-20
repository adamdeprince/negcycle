#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "common/types.h"

namespace nb = nanobind;

namespace {

template <typename T>
void hash_combine(std::size_t& seed, const T& value) {
  const std::size_t h = std::hash<T>{}(value);
  seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

struct SymbolPairHash {
  std::size_t operator()(const std::pair<std::string, std::string>& value) const noexcept {
    std::size_t seed = 0;
    hash_combine(seed, value.first);
    hash_combine(seed, value.second);
    return seed;
  }
};

bool edge_equal(const arbcycle::Edge& lhs, const arbcycle::Edge& rhs) {
  return lhs.from == rhs.from && lhs.to == rhs.to &&
         lhs.gross_rate == rhs.gross_rate && lhs.fee_bps == rhs.fee_bps &&
         lhs.net_rate == rhs.net_rate && lhs.weight == rhs.weight;
}

std::size_t edge_hash(const arbcycle::Edge& edge) {
  std::size_t seed = 0;
  hash_combine(seed, edge.from);
  hash_combine(seed, edge.to);
  hash_combine(seed, edge.gross_rate);
  hash_combine(seed, edge.fee_bps);
  hash_combine(seed, edge.net_rate);
  hash_combine(seed, edge.weight);
  return seed;
}

std::string edge_str(const arbcycle::Edge& edge) {
  std::ostringstream out;
  out << edge.from << " -> " << edge.to << " @ " << edge.net_rate;
  return out.str();
}

std::string edge_repr(const arbcycle::Edge& edge) {
  std::ostringstream out;
  out << "Edge(from=" << edge.from
      << ", to=" << edge.to
      << ", gross_rate=" << edge.gross_rate
      << ", fee_bps=" << edge.fee_bps
      << ", net_rate=" << edge.net_rate
      << ", weight=" << edge.weight << ")";
  return out.str();
}

bool cycle_equal(const arbcycle::Cycle& lhs, const arbcycle::Cycle& rhs) {
  if (lhs.vertices != rhs.vertices || lhs.names != rhs.names ||
      lhs.total_weight != rhs.total_weight || lhs.log_gain != rhs.log_gain ||
      lhs.gain_factor != rhs.gain_factor || lhs.pct_return != rhs.pct_return ||
      lhs.legs.size() != rhs.legs.size()) {
    return false;
  }

  for (std::size_t i = 0; i < lhs.legs.size(); ++i) {
    if (!edge_equal(lhs.legs[i], rhs.legs[i])) {
      return false;
    }
  }
  return true;
}

std::size_t cycle_hash(const arbcycle::Cycle& cycle) {
  std::size_t seed = 0;

  for (int vertex : cycle.vertices) {
    hash_combine(seed, vertex);
  }
  for (const std::string& name : cycle.names) {
    hash_combine(seed, name);
  }
  for (const arbcycle::Edge& leg : cycle.legs) {
    hash_combine(seed, leg.from);
    hash_combine(seed, leg.to);
    hash_combine(seed, leg.gross_rate);
    hash_combine(seed, leg.fee_bps);
    hash_combine(seed, leg.net_rate);
    hash_combine(seed, leg.weight);
  }

  hash_combine(seed, cycle.total_weight);
  hash_combine(seed, cycle.log_gain);
  hash_combine(seed, cycle.gain_factor);
  hash_combine(seed, cycle.pct_return);
  return seed;
}

std::unordered_set<std::pair<std::string, std::string>, SymbolPairHash> cycle_edge_set(
    const arbcycle::Cycle& cycle) {
  std::unordered_set<std::pair<std::string, std::string>, SymbolPairHash> out;
  out.reserve(cycle.legs.size());

  for (const arbcycle::Edge& leg : cycle.legs) {
    out.emplace(leg.from, leg.to);
  }

  return out;
}

bool cycle_same_path(const arbcycle::Cycle& lhs, const arbcycle::Cycle& rhs) {
  if (lhs.legs.size() != rhs.legs.size()) {
    return false;
  }

  return cycle_edge_set(lhs) == cycle_edge_set(rhs);
}

bool cycle_has_symbol(const arbcycle::Cycle& cycle, std::string_view symbol) {
  return std::find(cycle.names.begin(), cycle.names.end(), symbol) != cycle.names.end();
}

bool cycle_has_edge(const arbcycle::Cycle& cycle,
                    std::string_view from,
                    std::string_view to) {
  return std::any_of(
      cycle.legs.begin(),
      cycle.legs.end(),
      [&](const arbcycle::Edge& edge) { return edge.from == from && edge.to == to; });
}

using EdgeState = std::tuple<std::string, std::string, float, float, float, float>;
using CycleState = std::tuple<std::vector<int>,
                              std::vector<std::string>,
                              std::vector<arbcycle::Edge>,
                              float,
                              float,
                              float,
                              float>;

EdgeState edge_state(const arbcycle::Edge& edge) {
  return EdgeState{
      edge.from, edge.to, edge.gross_rate, edge.fee_bps, edge.net_rate, edge.weight};
}

void load_edge_state(arbcycle::Edge& edge, const EdgeState& state) {
  edge.from = std::get<0>(state);
  edge.to = std::get<1>(state);
  edge.gross_rate = std::get<2>(state);
  edge.fee_bps = std::get<3>(state);
  edge.net_rate = std::get<4>(state);
  edge.weight = std::get<5>(state);
}

CycleState cycle_state(const arbcycle::Cycle& cycle) {
  return CycleState{
      cycle.vertices,
      cycle.names,
      cycle.legs,
      cycle.total_weight,
      cycle.log_gain,
      cycle.gain_factor,
      cycle.pct_return};
}

void load_cycle_state(arbcycle::Cycle& cycle, const CycleState& state) {
  cycle.vertices = std::get<0>(state);
  cycle.names = std::get<1>(state);
  cycle.legs = std::get<2>(state);
  cycle.total_weight = std::get<3>(state);
  cycle.log_gain = std::get<4>(state);
  cycle.gain_factor = std::get<5>(state);
  cycle.pct_return = std::get<6>(state);
}

std::string cycle_path_string(const arbcycle::Cycle& cycle) {
  std::ostringstream out;

  if (!cycle.names.empty()) {
    for (std::size_t i = 0; i < cycle.names.size(); ++i) {
      if (i != 0) {
        out << " -> ";
      }
      out << cycle.names[i];
    }
    return out.str();
  }

  if (cycle.vertices.empty()) {
    return "[]";
  }

  out << "[";
  for (std::size_t i = 0; i < cycle.vertices.size(); ++i) {
    if (i != 0) {
      out << " -> ";
    }
    out << cycle.vertices[i];
  }
  out << "]";
  return out.str();
}

std::string cycle_str(const arbcycle::Cycle& cycle) {
  std::ostringstream out;
  out << cycle_path_string(cycle) << " (" << cycle.pct_return << "%)";
  return out.str();
}

std::string cycle_repr(const arbcycle::Cycle& cycle) {
  std::ostringstream out;
  out << "Cycle(length=" << cycle.length() << ", path=" << cycle_path_string(cycle)
      << ", gain_factor=" << cycle.gain_factor
      << ", pct_return=" << cycle.pct_return << ")";
  return out.str();
}

std::size_t wrap_index(Py_ssize_t i, std::size_t size) {
  if (i < 0) {
    i += static_cast<Py_ssize_t>(size);
  }
  if (i < 0 || static_cast<std::size_t>(i) >= size) {
    throw nb::index_error();
  }
  return static_cast<std::size_t>(i);
}

} // namespace

NB_MODULE(_common, m) {
  m.doc() = "Shared arbcycle nanobind types";

  nb::class_<arbcycle::Edge>(m, "Edge")
      .def(nb::init<>())
      .def("__len__", [](const arbcycle::Edge&) { return 2; })
      .def(
          "__getitem__",
          [](const arbcycle::Edge& self, Py_ssize_t i) -> const std::string& {
            switch (wrap_index(i, 2)) {
              case 0:
                return self.from;
              default:
                return self.to;
            }
          },
          nb::rv_policy::reference_internal)
      .def("__iter__", [](const arbcycle::Edge& self) {
        return nb::iter(nb::make_tuple(self.from, self.to));
      })
      .def("__reversed__", [](const arbcycle::Edge& self) {
        return nb::iter(nb::make_tuple(self.to, self.from));
      })
      .def("__contains__", [](const arbcycle::Edge& self, nb::handle item) {
        if (!nb::isinstance<nb::str>(item)) {
          return false;
        }

        const std::string symbol = nb::cast<std::string>(item);
        return self.from == symbol || self.to == symbol;
      })
      .def_ro("from_symbol", &arbcycle::Edge::from)
      .def_ro("to_symbol", &arbcycle::Edge::to)
      .def_ro("from", &arbcycle::Edge::from)
      .def_ro("to", &arbcycle::Edge::to)
      .def_ro("gross_rate", &arbcycle::Edge::gross_rate)
      .def_ro("fee_bps", &arbcycle::Edge::fee_bps)
      .def_ro("net_rate", &arbcycle::Edge::net_rate)
      .def_ro("weight", &arbcycle::Edge::weight)
      .def("__hash__", [](const arbcycle::Edge& self) { return edge_hash(self); })
      .def("__eq__", [](const arbcycle::Edge& lhs, const arbcycle::Edge& rhs) {
        return edge_equal(lhs, rhs);
      })
      .def("__getstate__", [](const arbcycle::Edge& self) { return edge_state(self); })
      .def("__setstate__", [](arbcycle::Edge& self, const EdgeState& state) {
        load_edge_state(self, state);
      })
      .def("__str__", [](const arbcycle::Edge& self) { return edge_str(self); })
      .def("__repr__", [](const arbcycle::Edge& self) { return edge_repr(self); });

  nb::class_<arbcycle::Cycle>(m, "Cycle")
      .def(nb::init<>())
      .def("__len__", [](const arbcycle::Cycle& self) { return self.legs.size(); })
      .def("__bool__", [](const arbcycle::Cycle&) { return true; })
      .def(
          "__getitem__",
          [](const arbcycle::Cycle& self, Py_ssize_t i) -> const arbcycle::Edge& {
            return self.legs[wrap_index(i, self.legs.size())];
          },
          nb::rv_policy::reference_internal)
      .def(
          "__iter__",
          [](arbcycle::Cycle& self) {
            return nb::make_iterator<nb::rv_policy::reference_internal>(
                nb::type<arbcycle::Cycle>(),
                "CycleLegIterator",
                self.legs.begin(),
                self.legs.end());
          },
          nb::keep_alive<0, 1>())
      .def(
          "__reversed__",
          [](arbcycle::Cycle& self) {
            return nb::make_iterator<nb::rv_policy::reference_internal>(
                nb::type<arbcycle::Cycle>(),
                "ReversedCycleLegIterator",
                self.legs.rbegin(),
                self.legs.rend());
          },
          nb::keep_alive<0, 1>())
      .def("__contains__", [](const arbcycle::Cycle& self, nb::handle item) {
        if (nb::isinstance<nb::str>(item)) {
          return cycle_has_symbol(self, nb::cast<std::string>(item));
        }

        if (nb::isinstance<nb::tuple>(item)) {
          nb::tuple t = nb::borrow<nb::tuple>(item);
          if (nb::len(t) == 2 && nb::isinstance<nb::str>(t[0]) &&
              nb::isinstance<nb::str>(t[1])) {
            return cycle_has_edge(
                self,
                nb::cast<std::string>(t[0]),
                nb::cast<std::string>(t[1]));
          }
          return false;
        }

        if (nb::isinstance<arbcycle::Edge>(item)) {
          const auto& edge = nb::cast<const arbcycle::Edge&>(item);
          return cycle_has_edge(self, edge.from, edge.to);
        }

        return false;
      })
      .def_prop_ro("length", &arbcycle::Cycle::length)
      .def_ro("vertices", &arbcycle::Cycle::vertices)
      .def_ro("names", &arbcycle::Cycle::names)
      .def_ro("legs", &arbcycle::Cycle::legs)
      .def_ro("total_weight", &arbcycle::Cycle::total_weight)
      .def_ro("log_gain", &arbcycle::Cycle::log_gain)
      .def_ro("gain_factor", &arbcycle::Cycle::gain_factor)
      .def_ro("pct_return", &arbcycle::Cycle::pct_return)
      .def("same", [](const arbcycle::Cycle& self, const arbcycle::Cycle& other) {
        return cycle_same_path(self, other);
      })
      .def("__hash__", [](const arbcycle::Cycle& self) { return cycle_hash(self); })
      .def("__eq__", [](const arbcycle::Cycle& lhs, const arbcycle::Cycle& rhs) {
        return cycle_equal(lhs, rhs);
      })
      .def("__getstate__", [](const arbcycle::Cycle& self) { return cycle_state(self); })
      .def("__setstate__", [](arbcycle::Cycle& self, const CycleState& state) {
        load_cycle_state(self, state);
      })
      .def("__str__", [](const arbcycle::Cycle& self) { return cycle_str(self); })
      .def("__repr__", [](const arbcycle::Cycle& self) { return cycle_repr(self); });
}
