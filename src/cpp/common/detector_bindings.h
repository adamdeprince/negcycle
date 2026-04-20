#pragma once

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/detector_base.h"

namespace arbcycle {

namespace nb = nanobind;
using namespace nb::literals;

namespace detail {

inline std::string_view trim_ascii(std::string_view value) {
  while (!value.empty() &&
         (value.front() == ' ' || value.front() == '\t' || value.front() == '\r' ||
          value.front() == '\n')) {
    value.remove_prefix(1);
  }

  while (!value.empty() &&
         (value.back() == ' ' || value.back() == '\t' || value.back() == '\r' ||
          value.back() == '\n')) {
    value.remove_suffix(1);
  }

  return value;
}

inline std::pair<std::string, std::string> parse_massive_currency_pair(std::string_view pair) {
  const std::size_t prefix_pos = pair.find(':');
  if (prefix_pos != std::string_view::npos) {
    pair.remove_prefix(prefix_pos + 1);
  }

  pair = trim_ascii(pair);

  const std::size_t split_pos = pair.find_first_of("-/");
  if (split_pos == std::string_view::npos || split_pos == 0 || split_pos + 1 >= pair.size()) {
    throw std::invalid_argument(
        "parse_massive_currency expects '<base>-<quote>' or '<base>/<quote>'");
  }

  return {
      std::string(trim_ascii(pair.substr(0, split_pos))),
      std::string(trim_ascii(pair.substr(split_pos + 1))),
  };
}

inline std::array<std::string_view, 6> parse_massive_csv_fields(std::string_view line) {
  line = trim_ascii(line);
  std::array<std::string_view, 6> fields{};

  std::size_t pos = 0;
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t comma = line.find(',', pos);
    if (comma == std::string_view::npos) {
      throw std::invalid_argument("massive currency row must contain at least 6 CSV fields");
    }

    fields[i] = trim_ascii(line.substr(pos, comma - pos));
    pos = comma + 1;
  }

  const std::size_t comma = line.find(',', pos);
  fields[5] = trim_ascii(
      comma == std::string_view::npos ? line.substr(pos) : line.substr(pos, comma - pos));
  return fields;
}

inline float parse_float_field(std::string_view field, const char* what) {
  const std::string value(field);
  char* end = nullptr;
  const float result = std::strtof(value.c_str(), &end);
  if (end == value.c_str() || (end != nullptr && *end != '\0')) {
    throw std::invalid_argument(std::string("invalid ") + what);
  }
  return result;
}

inline std::int64_t parse_i64_field(std::string_view field, const char* what) {
  const std::string value(field);
  char* end = nullptr;
  const long long result = std::strtoll(value.c_str(), &end, 10);
  if (end == value.c_str() || (end != nullptr && *end != '\0')) {
    throw std::invalid_argument(std::string("invalid ") + what);
  }
  return static_cast<std::int64_t>(result);
}

struct MassiveCurrencyQuoteRow {
  std::string from_symbol;
  std::string to_symbol;
  float ask_price{0.0f};
  float bid_price{0.0f};
  double timestamp_seconds{0.0};
};

inline std::vector<MassiveCurrencyQuoteRow> load_massive_currency_rows(nb::object file_obj) {
  std::vector<MassiveCurrencyQuoteRow> rows;
  nb::object readline = file_obj.attr("readline");
  std::size_t line_number = 0;

  for (;;) {
    const std::string line = nb::cast<std::string>(readline());
    if (line.empty()) {
      break;
    }

    ++line_number;
    const std::string_view trimmed = trim_ascii(line);
    if (trimmed.empty()) {
      continue;
    }

    const auto fields = parse_massive_csv_fields(trimmed);
    std::string from_symbol;
    std::string to_symbol;
    try {
      std::tie(from_symbol, to_symbol) = parse_massive_currency_pair(fields[0]);
    } catch (const std::invalid_argument&) {
      if (line_number == 1) {
        continue;
      }
      throw;
    }

    rows.push_back(MassiveCurrencyQuoteRow{
        .from_symbol = std::move(from_symbol),
        .to_symbol = std::move(to_symbol),
        .ask_price = parse_float_field(fields[2], "ask price"),
        .bid_price = parse_float_field(fields[4], "bid price"),
        .timestamp_seconds =
            static_cast<double>(parse_i64_field(fields[5], "participant timestamp")) * 1.0e-9,
    });
  }

  std::stable_sort(
      rows.begin(),
      rows.end(),
      [](const MassiveCurrencyQuoteRow& lhs, const MassiveCurrencyQuoteRow& rhs) {
        return lhs.timestamp_seconds < rhs.timestamp_seconds;
      });

  return rows;
}

template <typename Detector>
struct MassiveCurrencyBulkFileIterator {
  Detector* detector{nullptr};
  std::vector<MassiveCurrencyQuoteRow> rows;
  float fee_bps{0.0f};
  int max_cycle_length{5};
  std::size_t next_index{0};
};

template <typename Detector>
void bind_massive_currency_bulk_iterator(nb::module_& m, const char* internal_name) {
  using IteratorState = MassiveCurrencyBulkFileIterator<Detector>;

  if (nb::type<IteratorState>().is_valid()) {
    return;
  }

  const std::string iterator_name = std::string(internal_name) + "MassiveCurrencyBulkIterator";
  nb::class_<IteratorState>(m, iterator_name.c_str())
      .def("__iter__", [](nb::handle h) { return h; })
      .def("__next__", [](IteratorState& state) {
        while (state.next_index < state.rows.size()) {
          const MassiveCurrencyQuoteRow& row = state.rows[state.next_index++];
          auto cycle = state.detector->add_book_and_find_best_arbitrage(
              row.from_symbol,
              row.to_symbol,
              row.bid_price,
              row.ask_price,
              state.fee_bps,
              state.max_cycle_length);

          if (!cycle) {
            continue;
          }

          return std::make_tuple(
              *cycle,
              row.from_symbol,
              row.to_symbol,
              row.ask_price,
              row.bid_price,
              row.timestamp_seconds);
        }

        throw nb::stop_iteration();
      });
}

} // namespace detail

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
  auto parse_massive_currency = [](std::string_view pair) {
    const auto [from_symbol, to_symbol] = detail::parse_massive_currency_pair(pair);
    return std::make_tuple(from_symbol, to_symbol);
  };
  detail::bind_massive_currency_bulk_iterator<Detector>(m, internal_name);

  nb::class_<Detector>(m, internal_name)
      .def(nb::init<>())
      .def_static("parse_massive_currency", parse_massive_currency, nb::arg("pair"))
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
      .def("find_best_arbitrage",
           [](Detector& self, int max_cycle_length) {
             if (max_cycle_length < 3) {
               throw std::invalid_argument("max_cycle_length must be at least 3");
             }

             nb::gil_scoped_release release;
             return self.find_best_arbitrage(max_cycle_length);
           },
           nb::arg("max_cycle_length"))
      .def("add_quote_and_find_best_arbitrage",
           [](Detector& self,
              std::string_view from_code,
              std::string_view to_code,
              float executable_rate,
              float fee_bps,
              int max_cycle_length) {
             if (max_cycle_length < 3) {
               throw std::invalid_argument("max_cycle_length must be at least 3");
             }

             if (max_cycle_length > 3) {
               nb::gil_scoped_release release;
               return self.add_quote_and_find_best_arbitrage(
                   from_code, to_code, executable_rate, fee_bps, max_cycle_length);
             }

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
             if (max_cycle_length < 3) {
               throw std::invalid_argument("max_cycle_length must be at least 3");
             }

             if (max_cycle_length > 3) {
               nb::gil_scoped_release release;
               return self.add_book_and_find_best_arbitrage(
                   base, quote, bid, ask, fee_bps, max_cycle_length);
             }

             return self.add_book_and_find_best_arbitrage(
                 base, quote, bid, ask, fee_bps, max_cycle_length);
           },
           nb::arg("base"),
           nb::arg("quote"),
           nb::arg("bid"),
           nb::arg("ask"),
           nb::arg("fee_bps"),
           nb::arg("max_cycle_length"))
      .def(
          "process_massive_currency_bulk_file",
          [](Detector& self, nb::object file_obj, int max_cycle_length, float fee_bps) {
            using IteratorState = detail::MassiveCurrencyBulkFileIterator<Detector>;
            return nb::cast(IteratorState{
                .detector = &self,
                .rows = detail::load_massive_currency_rows(file_obj),
                .fee_bps = fee_bps,
                .max_cycle_length = max_cycle_length,
            });
          },
          nb::keep_alive<0, 1>(),
          nb::arg("file_obj"),
          nb::arg("max_cycle_length") = 5,
          nb::arg("fee_bps") = 0.0f)
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
