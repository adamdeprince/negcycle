#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <functional>
#include <sstream>
#include <string>

#include "common/types.h"

namespace nb = nanobind;

namespace {

template <typename T>
void hash_combine(std::size_t& seed, const T& value) {
  const std::size_t h = std::hash<T>{}(value);
  seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

bool edge_equal(const arbcycle::Edge& lhs, const arbcycle::Edge& rhs) {
  return lhs.from == rhs.from && lhs.to == rhs.to &&
         lhs.gross_rate == rhs.gross_rate && lhs.fee_bps == rhs.fee_bps &&
         lhs.net_rate == rhs.net_rate && lhs.weight == rhs.weight;
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
      .def_ro("from_symbol", &arbcycle::Edge::from)
      .def_ro("to_symbol", &arbcycle::Edge::to)
      .def_ro("from", &arbcycle::Edge::from)
      .def_ro("to", &arbcycle::Edge::to)
      .def_ro("gross_rate", &arbcycle::Edge::gross_rate)
      .def_ro("fee_bps", &arbcycle::Edge::fee_bps)
      .def_ro("net_rate", &arbcycle::Edge::net_rate)
      .def_ro("weight", &arbcycle::Edge::weight);

  nb::class_<arbcycle::Cycle>(m, "Cycle")
      .def("__len__", [](const arbcycle::Cycle& self) { return self.legs.size(); })
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
      .def_prop_ro("length", &arbcycle::Cycle::length)
      .def_ro("vertices", &arbcycle::Cycle::vertices)
      .def_ro("names", &arbcycle::Cycle::names)
      .def_ro("legs", &arbcycle::Cycle::legs)
      .def_ro("total_weight", &arbcycle::Cycle::total_weight)
      .def_ro("log_gain", &arbcycle::Cycle::log_gain)
      .def_ro("gain_factor", &arbcycle::Cycle::gain_factor)
      .def_ro("pct_return", &arbcycle::Cycle::pct_return)
      .def("__hash__", [](const arbcycle::Cycle& self) { return cycle_hash(self); })
      .def("__eq__", [](const arbcycle::Cycle& lhs, const arbcycle::Cycle& rhs) {
        return cycle_equal(lhs, rhs);
      })
      .def("__str__", [](const arbcycle::Cycle& self) { return cycle_str(self); })
      .def("__repr__", [](const arbcycle::Cycle& self) { return cycle_repr(self); });
}
