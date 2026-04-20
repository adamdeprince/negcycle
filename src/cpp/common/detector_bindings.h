#pragma once

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <sstream>
#include <string>

#include "common/detector_base.h"

namespace arbcycle {

namespace nb = nanobind;
using namespace nb::literals;

template <typename Detector>
void bind_detector_module(nb::module_& m, const char* internal_name, const char* doc) {
  using QuoteState = std::tuple<std::string, std::string, float, float>;
  using DetectorState = std::tuple<std::vector<std::string>, std::vector<QuoteState>>;

  nb::module_ common = nb::module_::import_("arbcycle._common");
  m.doc() = doc;
  auto serialize_state = [](const Detector& self) -> DetectorState {
    std::vector<QuoteState> quotes;
    const auto serialized = self.serialized_quotes();
    quotes.reserve(serialized.size());

    for (const auto& q : serialized) {
      quotes.emplace_back(q.from, q.to, q.gross_rate, q.fee_bps);
    }

    return DetectorState{self.currencies(), std::move(quotes)};
  };
  auto detector_str = [](const Detector& self) {
    std::ostringstream out;
    out << "ArbitrageDetector("
        << self.currencies().size() << " currencies, "
        << self.serialized_quotes().size() << " quotes)";
    return out.str();
  };
  auto detector_repr = [internal_name](const Detector& self) {
    std::ostringstream out;
    out << internal_name
        << "(currencies=" << self.currencies().size()
        << ", quotes=" << self.serialized_quotes().size() << ")";
    return out.str();
  };
  auto restore_from_state = [](Detector& self, const DetectorState& state) {
    std::vector<ArbitrageDetectorBase::SerializedQuote> quotes;
    const auto& quote_state = std::get<1>(state);
    quotes.reserve(quote_state.size());

    for (const auto& q : quote_state) {
      quotes.push_back(ArbitrageDetectorBase::SerializedQuote{
          .from = std::get<0>(q),
          .to = std::get<1>(q),
          .gross_rate = std::get<2>(q),
          .fee_bps = std::get<3>(q),
      });
    }

    self.restore_state(std::get<0>(state), quotes);
  };
  auto make_copy = [](const Detector& self) {
    return Detector(self);
  };

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
      })
      .def("__len__", [](const Detector& self) { return self.currencies().size(); })
      .def("__bool__", [](const Detector& self) { return !self.currencies().empty(); })
      .def(
          "__iter__",
          [](Detector& self) {
            return nb::make_iterator<nb::rv_policy::copy>(
                nb::type<Detector>(),
                "DetectorCurrencyIterator",
                self.currencies().begin(),
                self.currencies().end());
          },
          nb::keep_alive<0, 1>())
      .def("__contains__", [](const Detector& self, nb::handle item) {
        if (nb::isinstance<nb::str>(item)) {
          return self.has_currency(nb::cast<std::string>(item));
        }

        if (nb::isinstance<nb::tuple>(item)) {
          nb::tuple t = nb::borrow<nb::tuple>(item);
          if (nb::len(t) == 2 && nb::isinstance<nb::str>(t[0]) &&
              nb::isinstance<nb::str>(t[1])) {
            return self.has_quote(
                nb::cast<std::string>(t[0]),
                nb::cast<std::string>(t[1]));
          }
          return false;
        }

        if (nb::isinstance<arbcycle::Edge>(item)) {
          const auto& edge = nb::cast<const arbcycle::Edge&>(item);
          return self.has_quote(edge.from, edge.to);
        }

        return false;
      })
      .def("__str__", detector_str)
      .def("__repr__", detector_repr)
      .def("__copy__", make_copy)
      .def("__deepcopy__", [make_copy](const Detector& self, nb::handle) {
        return make_copy(self);
      })
      .def("__getstate__", serialize_state)
      .def("__setstate__", restore_from_state)
      .def("__reduce__", [serialize_state](const Detector& self) {
        nb::object restore = nb::module_::import_("arbcycle").attr("_restore_detector");
        return nb::make_tuple(std::move(restore), nb::make_tuple(serialize_state(self)));
      });

  m.attr("Edge") = common.attr("Edge");
  m.attr("Cycle") = common.attr("Cycle");
  m.attr("ArbitrageDetector") = m.attr(internal_name);
}

} // namespace arbcycle
