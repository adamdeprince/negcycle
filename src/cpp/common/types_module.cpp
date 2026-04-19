#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "common/types.h"

namespace nb = nanobind;

NB_MODULE(_common, m) {
  m.doc() = "Shared arbcycle nanobind types";

  nb::class_<arbcycle::Edge>(m, "Edge")
      .def_ro("from_id", &arbcycle::Edge::from)
      .def_ro("to_id", &arbcycle::Edge::to)
      .def_ro("gross_rate", &arbcycle::Edge::gross_rate)
      .def_ro("fee_bps", &arbcycle::Edge::fee_bps)
      .def_ro("net_rate", &arbcycle::Edge::net_rate)
      .def_ro("weight", &arbcycle::Edge::weight);

  nb::class_<arbcycle::Cycle>(m, "Cycle")
      .def_prop_ro("length", &arbcycle::Cycle::length)
      .def_ro("vertices", &arbcycle::Cycle::vertices)
      .def_ro("names", &arbcycle::Cycle::names)
      .def_ro("legs", &arbcycle::Cycle::legs)
      .def_ro("total_weight", &arbcycle::Cycle::total_weight)
      .def_ro("log_gain", &arbcycle::Cycle::log_gain)
      .def_ro("gain_factor", &arbcycle::Cycle::gain_factor)
      .def_ro("pct_return", &arbcycle::Cycle::pct_return);
}
