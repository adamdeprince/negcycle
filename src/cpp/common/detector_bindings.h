#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include "common/detector_base.h"

namespace arbcycle {

namespace nb = nanobind;
using namespace nb::literals;

template <typename Detector>
void bind_detector_module(nb::module_& m, const char* internal_name, const char* doc) {
  nb::module_ common = nb::module_::import_("arbcycle._common");
  m.doc() = doc;

  nb::class_<Detector>(m, internal_name)
      .def(nb::init<>())
      .def("add_currency", [](Detector& self, std::string_view code) {
        return self.add_currency(code);
      }, nb::arg("code"))
      .def("add_quote",
           [](Detector& self,
              std::string_view from_code,
              std::string_view to_code,
              float executable_rate,
              float fee_bps) {
             self.add_quote(from_code, to_code, executable_rate, fee_bps);
           },
           nb::arg("from_code"),
           nb::arg("to_code"),
           nb::arg("executable_rate"),
           nb::arg("fee_bps") = 0.0f)
      .def("add_book",
           [](Detector& self,
              std::string_view base,
              std::string_view quote,
              float bid,
              float ask,
              float fee_bps) {
             self.add_book(base, quote, bid, ask, fee_bps);
           },
           nb::arg("base"),
           nb::arg("quote"),
           nb::arg("bid"),
           nb::arg("ask"),
           nb::arg("fee_bps") = 0.0f)
      .def("find_best_arbitrage", [](Detector& self, int max_cycle_length) {
        return self.find_best_arbitrage(max_cycle_length);
      }, nb::arg("max_cycle_length"))
      .def("add_quote_and_find_best_arbitrage",
           [](Detector& self,
              std::string_view from_code,
              std::string_view to_code,
              float executable_rate,
              float fee_bps,
              int max_cycle_length) {
             return self.add_quote_and_find_best_arbitrage(
                 from_code, to_code, executable_rate, fee_bps, max_cycle_length);
           },
           nb::arg("from_code"),
           nb::arg("to_code"),
           nb::arg("executable_rate"),
           nb::arg("fee_bps"),
           nb::arg("max_cycle_length"))
      .def("add_book_and_find_best_arbitrage",
           [](Detector& self,
              std::string_view base,
              std::string_view quote,
              float bid,
              float ask,
              float fee_bps,
              int max_cycle_length) {
             return self.add_book_and_find_best_arbitrage(
                 base, quote, bid, ask, fee_bps, max_cycle_length);
           },
           nb::arg("base"),
           nb::arg("quote"),
           nb::arg("bid"),
           nb::arg("ask"),
           nb::arg("fee_bps"),
           nb::arg("max_cycle_length"))
      .def_prop_ro("currencies", [](const Detector& self) {
        return self.currencies();
      });

  m.attr("Edge") = common.attr("Edge");
  m.attr("Cycle") = common.attr("Cycle");
  m.attr("ArbitrageDetector") = m.attr(internal_name);
}

} // namespace arbcycle
