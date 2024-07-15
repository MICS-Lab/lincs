// Copyright 2023-2024 Vincent Jacques

#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>

#include "../io.hpp"
#include "../vendored/magic_enum.hpp"
#include "../vendored/pybind11/pybind11.h"
#include "../vendored/pybind11/operators.h"
#include "../vendored/pybind11/stl.h"


namespace py = pybind11;
using namespace pybind11::literals;

namespace {

class PythonOutputDevice : public boost::iostreams::sink {
 public:
  explicit PythonOutputDevice(py::object out_file_) : out_file(out_file_) {}

  std::streamsize write(const char* s, std::streamsize n) {
    out_file.attr("write")(std::string(s, n));
    return n;
  }

 private:
  py::object out_file;
};

void dump_problem(const lincs::Problem& problem, py::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  problem.dump(out_stream);
}

void dump_model(const lincs::Model& model, const lincs::Problem& problem, py::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  model.dump(problem, out_stream);
}

void dump_alternatives(const lincs::Alternatives& alternatives, const lincs::Problem& problem, py::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  alternatives.dump(problem, out_stream);
}

class PythonInputDevice : public boost::iostreams::source {
 public:
  explicit PythonInputDevice(py::object in_file_) : in_file(in_file_) {}

  std::streamsize read(char* s, std::streamsize n) {
    std::string str = in_file.attr("read")(n).cast<std::string>();
    std::copy(str.begin(), str.end(), s);
    return str.size();
  }

 private:
  py::object in_file;
};

lincs::Problem load_problem(py::object& in_file) {
  boost::iostreams::stream<PythonInputDevice> in_stream(in_file);
  return lincs::Problem::load(in_stream);
}

lincs::Model load_model(const lincs::Problem& problem, py::object& in_file) {
  boost::iostreams::stream<PythonInputDevice> in_stream(in_file);
  return lincs::Model::load(problem, in_stream);
}

lincs::Alternatives load_alternatives(const lincs::Problem& problem, py::object& in_file) {
  boost::iostreams::stream<PythonInputDevice> in_stream(in_file);
  return lincs::Alternatives::load(problem, in_stream);
}

template <typename T, typename Scope>
auto auto_enum(Scope& scope, const char* name, const char* docstring = nullptr) {
  auto e = py::enum_<T>(scope, name, docstring);
  for(T value : magic_enum::enum_values<T>()) {
    e.value(std::string(magic_enum::enum_name(value)).c_str(), value);
  }
  return e;
}

// Inspired by https://stackoverflow.com/a/57643845/905845
template<typename V, std::size_t I=0>
V variant_cast(const py::object& obj) {
  if constexpr (I < std::variant_size_v<V>) {
    try {
      return obj.cast<std::variant_alternative_t<I, V>>();
    } catch (py::cast_error&) {
      return variant_cast<V, I + 1>(obj);
    }
  }
  throw py::cast_error();
}

}  // namespace

namespace lincs {

void define_problem_classes(py::module& m) {
  auto criterion_class = py::class_<lincs::Criterion>(
    m,
    "Criterion",
    "A classification criterion, to be used in a classification :py:class:`Problem`."
  )
    .def_property_readonly("name", &lincs::Criterion::get_name, "The name of the criterion.")
    .def_property_readonly("value_type", &lincs::Criterion::get_value_type, "The type of values for this criterion.")
    .def_property_readonly("is_real", &lincs::Criterion::is_real, "``True`` if the criterion is real-valued.")
    .def_property_readonly("is_integer", &lincs::Criterion::is_integer, "``True`` if the criterion is integer-valued.")
    .def_property_readonly("is_enumerated", &lincs::Criterion::is_enumerated, "``True`` if the criterion takes enumerated values.")
    // @todo(Project management, later) Expose all our 'std::variant's as 'Union's; this is easy now that we use pybind11
    // @todo(Project management, even later, when we drop Python 3.9) Deprecate the discriminated accessors like below, and rely on on 'match/case' (available since Python 3.10).
    .def_property_readonly("real_values", &lincs::Criterion::get_real_values, "Descriptor of the real values allowed for this criterion, accessible if ``is_real``.")
    .def_property_readonly("integer_values", &lincs::Criterion::get_integer_values, "Descriptor of the integer values allowed for this criterion, accessible if ``is_integer``.")
    .def_property_readonly("enumerated_values", &lincs::Criterion::get_enumerated_values, "Descriptor of the enumerated values allowed for this criterion, accessible if ``is_enumerated``.")
    .def(py::pickle(
      [](const lincs::Criterion& criterion) {
        return std::visit(
          [&criterion](const auto& values) { return py::make_tuple(criterion.get_name(), values); },
          criterion.get_values()
        );
      },
      [](py::tuple t) {
        return lincs::Criterion(t[0].cast<std::string>(), variant_cast<lincs::Criterion::Values>(t[1]));
      }
    ))
    .def(py::self == py::self)  // Private, undocumented, used only for our tests
  ;

  auto_enum<lincs::Criterion::ValueType>(
    criterion_class,
    "ValueType",
    "The different types of values for a criterion."
  );

  auto_enum<lincs::Criterion::PreferenceDirection>(
    criterion_class,
    "PreferenceDirection",
    "What values are preferred for a criterion."
  )
    .value("isotone", lincs::Criterion::PreferenceDirection::isotone)
    .value("antitone", lincs::Criterion::PreferenceDirection::antitone)
  ;

  py::class_<lincs::Criterion::RealValues>(
    criterion_class,
    "RealValues",
    "Descriptor of the real values allowed for a criterion."
  )
    .def(
      py::init<lincs::Criterion::PreferenceDirection, float, float>(),
      "preference_direction"_a, "min_value"_a, "max_value"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("preference_direction", &lincs::Criterion::RealValues::get_preference_direction, "The preference direction for this criterion.")
    .def_property_readonly("is_increasing", &lincs::Criterion::RealValues::is_increasing, "``True`` if the criterion has increasing preference direction.")
    .def_property_readonly("is_decreasing", &lincs::Criterion::RealValues::is_decreasing, "``True`` if the criterion has decreasing preference direction.")
    .def_property_readonly("is_single_peaked", &lincs::Criterion::RealValues::is_single_peaked, "``True`` if the criterion has single-peaked preference direction.")
    .def_property_readonly("min_value", &lincs::Criterion::RealValues::get_min_value, "The minimum value allowed for this criterion.")
    .def_property_readonly("max_value", &lincs::Criterion::RealValues::get_max_value, "The maximum value allowed for this criterion.")
    .def(py::pickle(
      [](const lincs::Criterion::RealValues& values) {
        return py::make_tuple(values.get_preference_direction(), values.get_min_value(), values.get_max_value());
      },
      [](py::tuple t) {
        return lincs::Criterion::RealValues(t[0].cast<lincs::Criterion::PreferenceDirection>(), t[1].cast<float>(), t[2].cast<float>());
      }
    ))
  ;

  py::class_<lincs::Criterion::IntegerValues>(
    criterion_class,
    "IntegerValues",
    "Descriptor of the integer values allowed for a criterion."
  )
    .def(
      py::init<lincs::Criterion::PreferenceDirection, int, int>(),
      "preference_direction"_a, "min_value"_a, "max_value"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("preference_direction", &lincs::Criterion::IntegerValues::get_preference_direction, "The preference direction for this criterion.")
    .def_property_readonly("is_increasing", &lincs::Criterion::IntegerValues::is_increasing, "``True`` if the criterion has increasing preference direction.")
    .def_property_readonly("is_decreasing", &lincs::Criterion::IntegerValues::is_decreasing, "``True`` if the criterion has decreasing preference direction.")
    .def_property_readonly("is_single_peaked", &lincs::Criterion::IntegerValues::is_single_peaked, "``True`` if the criterion has single-peaked preference direction.")
    .def_property_readonly("min_value", &lincs::Criterion::IntegerValues::get_min_value, "The minimum value allowed for this criterion.")
    .def_property_readonly("max_value", &lincs::Criterion::IntegerValues::get_max_value, "The maximum value allowed for this criterion.")
    .def(py::pickle(
      [](const lincs::Criterion::IntegerValues& values) {
        return py::make_tuple(values.get_preference_direction(), values.get_min_value(), values.get_max_value());
      },
      [](py::tuple t) {
        return lincs::Criterion::IntegerValues(t[0].cast<lincs::Criterion::PreferenceDirection>(), t[1].cast<int>(), t[2].cast<int>());
      }
    ))
  ;

  criterion_class.attr("EnumeratedValues") = py::class_<lincs::Criterion::EnumeratedValues>(
    criterion_class,
    "EnumeratedValues",
    "Descriptor of the enumerated values allowed for a criterion."
  )
    .def(py::init<std::vector<std::string>>(),
      "ordered_values"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("ordered_values", &lincs::Criterion::EnumeratedValues::get_ordered_values, "The values for this criterion, from the worst to the best.")
    .def("get_value_rank", &lincs::Criterion::EnumeratedValues::get_value_rank, "value"_a, "Get the rank of a given value.")
    .def(py::pickle(
      [](const lincs::Criterion::EnumeratedValues& values) {
        return py::make_tuple(values.get_ordered_values());
      },
      [](py::tuple t) {
        return lincs::Criterion::EnumeratedValues(t[0].cast<std::vector<std::string>>());
      }
    ))
  ;

  criterion_class
    .def(
      py::init<std::string, lincs::Criterion::RealValues>(),
      "name"_a, "values"_a,
      "Constructor for real-valued criterion."
    )
    .def(
      py::init<std::string, lincs::Criterion::IntegerValues>(),
      "name"_a, "values"_a,
      "Constructor for integer-valued criterion."
    )
    .def(
      py::init<std::string, lincs::Criterion::EnumeratedValues>(),
      "name"_a, "values"_a,
      "Constructor for criterion with enumerated values."
    )
  ;

  py::class_<lincs::Category>(
    m,
    "Category",
    "A category of a classification :py:class:`Problem`."
  )
    .def(
      py::init<std::string>(),
      "name"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("name", &lincs::Category::get_name, "The name of this category.")
    .def(py::pickle(
      [](const lincs::Category& category) {
        return py::make_tuple(category.get_name());
      },
      [](py::tuple t) {
        return lincs::Category(t[0].cast<std::string>());
      }
    ))
  ;

  auto problem_class = py::class_<lincs::Problem>(
    m,
    "Problem",
    "A classification problem, with criteria and categories."
  )
    .def(
      py::init<std::vector<lincs::Criterion>, std::vector<lincs::Category>>(),
      "criteria"_a, "ordered_categories"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("criteria", &lincs::Problem::get_criteria, "The criteria of this problem.")
    .def_property_readonly("ordered_categories", &lincs::Problem::get_ordered_categories, "The categories of this problem, from the worst to the best.")
    .def(
      "dump",
      &dump_problem,
      "out"_a,
      "Dump the problem to the provided ``.write``-supporting file-like object, in YAML format."
    )
    .def_static(
      "load",
      &load_problem,
      "in"_a,
      "Load a problem from the provided ``.read``-supporting file-like object, in YAML format."
    )
    .def(py::pickle(
      [](const lincs::Problem& problem) {
        return py::make_tuple(
          problem.get_criteria(),
          problem.get_ordered_categories()
        );
      },
      [](py::tuple t) {
        return lincs::Problem(
          t[0].cast<std::vector<lincs::Criterion>>(),
          t[1].cast<std::vector<lincs::Category>>()
        );
      }
    ))
  ;
  problem_class.attr("JSON_SCHEMA") = lincs::Problem::json_schema;
}

void define_model_classes(py::module& m) {
  auto accepted_values_class = py::class_<lincs::AcceptedValues>(
    m,
    "AcceptedValues",
    "The values accepted by a model for a criterion."
  )
    .def_property_readonly("value_type", &lincs::AcceptedValues::get_value_type, "The type of values for the corresponding criterion.")
    .def_property_readonly("is_real", &lincs::AcceptedValues::is_real, "``True`` if the corresponding criterion is real-valued.")
    .def_property_readonly("is_integer", &lincs::AcceptedValues::is_integer, "``True`` if the corresponding criterion is integer-valued.")
    .def_property_readonly("is_enumerated", &lincs::AcceptedValues::is_enumerated, "``True`` if the corresponding criterion takes enumerated values.")
    .def_property_readonly("kind", &lincs::AcceptedValues::get_kind, "The kind of descriptor for these accepted values.")
    .def_property_readonly("is_thresholds", &lincs::AcceptedValues::is_thresholds, "``True`` if the descriptor is a set of thresholds.")
    .def_property_readonly("is_intervals", &lincs::AcceptedValues::is_intervals, "``True`` if the descriptor is a set of intervals.")
    .def_property_readonly("real_thresholds", &lincs::AcceptedValues::get_real_thresholds, "Descriptor of the real thresholds, accessible if ``is_real and is_thresholds``.")
    .def_property_readonly("integer_thresholds", &lincs::AcceptedValues::get_integer_thresholds, "Descriptor of the integer thresholds, accessible if ``is_integer and is_thresholds``.")
    .def_property_readonly("enumerated_thresholds", &lincs::AcceptedValues::get_enumerated_thresholds, "Descriptor of the enumerated thresholds, accessible if ``is_enumerated and is_thresholds``.")
    .def_property_readonly("real_intervals", &lincs::AcceptedValues::get_real_intervals, "Descriptor of the real intervals, accessible if ``is_real and is_intervals``.")
    .def_property_readonly("integer_intervals", &lincs::AcceptedValues::get_integer_intervals, "Descriptor of the integer intervals, accessible if ``is_integer and is_intervals``.")
    .def(py::pickle(
      [](const lincs::AcceptedValues& accepted_values) {
        return std::visit(
          [&accepted_values](const auto& thresholds) { return py::make_tuple(thresholds); },
          accepted_values.get()
        );
      },
      [](py::tuple t) {
        return lincs::AcceptedValues(variant_cast<lincs::AcceptedValues::Self>(t[0]));
      }
    ))
  ;

  auto_enum<lincs::AcceptedValues::Kind>(
    accepted_values_class,
    "Kind",
    "The different kinds of descriptors for accepted values."
  );

  py::class_<lincs::AcceptedValues::RealThresholds>(
    accepted_values_class,
    "RealThresholds",
    "Descriptor for thresholds for an real-valued criterion."
  )
    .def(
      py::init<const std::vector<std::optional<float>>&>(),
      "thresholds"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("thresholds", &lincs::AcceptedValues::RealThresholds::get_thresholds, "The thresholds for this descriptor.")
    .def(py::pickle(
      [](const lincs::AcceptedValues::RealThresholds& thresholds) {
        return py::make_tuple(thresholds.get_thresholds());
      },
      [](py::tuple t) {
        return lincs::AcceptedValues::RealThresholds(t[0].cast<std::vector<std::optional<float>>>());
      }
    ))
  ;

  py::class_<lincs::AcceptedValues::IntegerThresholds>(
    accepted_values_class,
    "IntegerThresholds",
    "Descriptor for thresholds for an integer-valued criterion."
  )
    .def(
      py::init<const std::vector<std::optional<int>>&>(),
      "thresholds"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("thresholds", &lincs::AcceptedValues::IntegerThresholds::get_thresholds, "The thresholds for this descriptor.")
    .def(py::pickle(
      [](const lincs::AcceptedValues::IntegerThresholds& thresholds) {
        return py::make_tuple(thresholds.get_thresholds());
      },
      [](py::tuple t) {
        return lincs::AcceptedValues::IntegerThresholds(t[0].cast<std::vector<std::optional<int>>>());
      }
    ))
  ;

  py::class_<lincs::AcceptedValues::EnumeratedThresholds>(
    accepted_values_class,
    "EnumeratedThresholds",
    "Descriptor for thresholds for a criterion taking enumerated values."
  )
    .def(
      py::init<const std::vector<std::optional<std::string>>&>(),
      "thresholds"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("thresholds", &lincs::AcceptedValues::EnumeratedThresholds::get_thresholds, "The thresholds for this descriptor.")
    .def(py::pickle(
      [](const lincs::AcceptedValues::EnumeratedThresholds& thresholds) {
        return py::make_tuple(thresholds.get_thresholds());
      },
      [](py::tuple t) {
        return lincs::AcceptedValues::EnumeratedThresholds(t[0].cast<std::vector<std::optional<std::string>>>());
      }
    ))
  ;

  py::class_<lincs::AcceptedValues::RealIntervals>(
    accepted_values_class,
    "RealIntervals",
    "Descriptor for intervals for an real-valued criterion."
  )
    .def(
      py::init<const std::vector<std::optional<std::pair<float, float>>>&>(),
      "intervals"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("intervals", &lincs::AcceptedValues::RealIntervals::get_intervals, "The intervals for this descriptor.")
    .def(py::pickle(
      [](const lincs::AcceptedValues::RealIntervals& intervals) {
        return py::make_tuple(intervals.get_intervals());
      },
      [](py::tuple t) {
        return lincs::AcceptedValues::RealIntervals(t[0].cast<std::vector<std::optional<std::pair<float, float>>>>());
      }
    ))
  ;

  py::class_<lincs::AcceptedValues::IntegerIntervals>(
    accepted_values_class,
    "IntegerIntervals",
    "Descriptor for intervals for an integer-valued criterion."
  )
    .def(
      py::init<const std::vector<std::optional<std::pair<int, int>>>&>(),
      "intervals"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("intervals", &lincs::AcceptedValues::IntegerIntervals::get_intervals, "The intervals for this descriptor.")
    .def(py::pickle(
      [](const lincs::AcceptedValues::IntegerIntervals& intervals) {
        return py::make_tuple(intervals.get_intervals());
      },
      [](py::tuple t) {
        return lincs::AcceptedValues::IntegerIntervals(t[0].cast<std::vector<std::optional<std::pair<int, int>>>>());
      }
    ))
  ;

  accepted_values_class
    .def(
      py::init<lincs::AcceptedValues::RealThresholds>(),
      "values"_a,
      "Constructor for thresholds on a real-valued criterion."
    )
    .def(
      py::init<lincs::AcceptedValues::IntegerThresholds>(),
      "values"_a,
      "Constructor for thresholds on an integer-valued criterion."
    )
    .def(
      py::init<lincs::AcceptedValues::EnumeratedThresholds>(),
      "values"_a,
      "Constructor for thresholds on an enumerated criterion."
    )
    .def(
      py::init<lincs::AcceptedValues::RealIntervals>(),
      "values"_a,
      "Constructor for intervals on a real-valued criterion."
    )
    .def(
      py::init<lincs::AcceptedValues::IntegerIntervals>(),
      "values"_a,
      "Constructor for intervals on an integer-valued criterion."
    )
  ;

  auto sufficient_coalitions_class = py::class_<lincs::SufficientCoalitions>(
    m,
    "SufficientCoalitions",
    "The coalitions of sufficient criteria to accept an alternative in a category."
  )
    .def_property_readonly("kind", &lincs::SufficientCoalitions::get_kind, "The kind of descriptor for these sufficient coalitions.")
    .def_property_readonly("is_weights", &lincs::SufficientCoalitions::is_weights, "``True`` if the descriptor is a set of weights.")
    .def_property_readonly("is_roots", &lincs::SufficientCoalitions::is_roots, "``True`` if the descriptor is a set of roots.")
    .def_property_readonly("weights", &lincs::SufficientCoalitions::get_weights, "Descriptor of the weights, accessible if ``is_weights``.")
    .def_property_readonly("roots", &lincs::SufficientCoalitions::get_roots, "Descriptor of the roots, accessible if ``is_roots``.")
    .def(py::pickle(
      [](const lincs::SufficientCoalitions& sufficient_coalitions) {
        return std::visit(
          [&sufficient_coalitions](const auto& descriptor) { return py::make_tuple(descriptor); },
          sufficient_coalitions.get()
        );
      },
      [](py::tuple t) {
        return lincs::SufficientCoalitions(variant_cast<lincs::SufficientCoalitions::Self>(t[0]));
      }
    ))
    .def(py::self == py::self)
  ;

  auto_enum<lincs::SufficientCoalitions::Kind>(
    sufficient_coalitions_class,
    "Kind",
    "The different kinds of descriptors for sufficient coalitions."
  );

  py::class_<lincs::SufficientCoalitions::Weights>(
    sufficient_coalitions_class,
    "Weights",
    "Descriptor for sufficient coalitions defined by weights."
  )
    .def(
      py::init<const std::vector<float>&>(),
      "criterion_weights"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("criterion_weights", &lincs::SufficientCoalitions::Weights::get_criterion_weights, "The weights for each criterion.")
    .def(py::pickle(
      [](const lincs::SufficientCoalitions::Weights& weights) {
        return py::make_tuple(weights.get_criterion_weights());
      },
      [](py::tuple t) {
        return lincs::SufficientCoalitions::Weights(t[0].cast<std::vector<float>>());
      }
    ))
  ;

  py::class_<lincs::SufficientCoalitions::Roots>(
    sufficient_coalitions_class,
    "Roots",
    "Descriptor for sufficient coalitions defined by roots."
  )
    .def(
      py::init<const Problem&, const std::vector<std::vector<unsigned>>&>(),
      "problem"_a, "upset_roots"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("upset_roots", &lincs::SufficientCoalitions::Roots::get_upset_roots_as_vectors, "The roots of the upset of sufficient coalitions.")
    .def(py::pickle(
      [](const lincs::SufficientCoalitions::Roots& roots) {
        const auto& bitsets = roots.get_upset_roots_as_bitsets();
        if (bitsets.empty()) {
          return py::make_tuple(0, std::vector<std::vector<unsigned>>());
        } else {
          return py::make_tuple(bitsets[0].size(), roots.get_upset_roots_as_vectors());
        }
      },
      [](py::tuple t) {
        return lincs::SufficientCoalitions::Roots(
          lincs::Internal(),
          t[0].cast<unsigned>(),
          t[1].cast<std::vector<std::vector<unsigned>>>()
        );
      }
    ))
  ;

  sufficient_coalitions_class
    .def(
      py::init<lincs::SufficientCoalitions::Weights>(),
      "weights"_a,
      "Constructor for sufficient coalitions defined by weights."
    )
    .def(
      py::init<lincs::SufficientCoalitions::Roots>(),
      "roots"_a,
      "Constructor for sufficient coalitions defined by roots."
    )
  ;

  auto model_class = py::class_<lincs::Model>(
    m,
    "Model",
    "An NCS classification model."
  )
    .def(
      py::init<const lincs::Problem&, const std::vector<lincs::AcceptedValues>&, const std::vector<lincs::SufficientCoalitions>&>(),
      "problem"_a, "accepted_values"_a, "sufficient_coalitions"_a,
      "The :py:class:`Model` being initialized must correspond to the given :py:class:`Problem`. Other parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("accepted_values", &lincs::Model::get_accepted_values, "The accepted values for each criterion.")
    .def_property_readonly("sufficient_coalitions", &lincs::Model::get_sufficient_coalitions, "The sufficient coalitions for each category.")
    .def(
      "dump",
      &dump_model,
      "problem"_a, "out"_a,
      "Dump the model to the provided ``.write``-supporting file-like object, in YAML format."
    )
    .def_static(
      "load",
      &load_model,
      "problem"_a, "in"_a,
      "Load a model for the provided ``Problem``, from the provided ``.read``-supporting file-like object, in YAML format."
    )
    .def(py::pickle(
      [](const lincs::Model& model) {
        return py::make_tuple(
          model.get_accepted_values(),
          model.get_sufficient_coalitions()
        );
      },
      [](py::tuple t) {
        return lincs::Model(
          lincs::Internal(),
          t[0].cast<std::vector<lincs::AcceptedValues>>(),
          t[1].cast<std::vector<lincs::SufficientCoalitions>>()
        );
      }
    ))
  ;
  model_class.attr("JSON_SCHEMA") = lincs::Model::json_schema;
}

void define_alternative_classes(py::module& m) {
  auto performance_class = py::class_<lincs::Performance>(
    m,
    "Performance",
    "The performance of an alternative on a criterion."
  )
    .def_property_readonly("value_type", &lincs::Performance::get_value_type, "The type of values for the corresponding criterion.")
    .def_property_readonly("is_real", &lincs::Performance::is_real, "``True`` if the corresponding criterion is real-valued.")
    .def_property_readonly("is_integer", &lincs::Performance::is_integer, "``True`` if the corresponding criterion is integer-valued.")
    .def_property_readonly("is_enumerated", &lincs::Performance::is_enumerated, "``True`` if the corresponding criterion takes enumerated values.")
    .def_property_readonly("real", &lincs::Performance::get_real, "The real performance, accessible if ``is_real``.")
    .def_property_readonly("integer", &lincs::Performance::get_integer, "The integer performance, accessible if ``is_integer``.")
    .def_property_readonly("enumerated", &lincs::Performance::get_enumerated, "The enumerated performance, accessible if ``is_enumerated``.")
    .def(py::pickle(
      [](const lincs::Performance& performance) {
        return std::visit(
          [&performance](const auto& perf) { return py::make_tuple(perf); },
          performance.get()
        );
      },
      [](py::tuple t) {
        return lincs::Performance(variant_cast<lincs::Performance::Self>(t[0]));
      }
    ))
  ;

  py::class_<lincs::Performance::Real>(
    performance_class,
    "Real",
    "A performance for a real-valued criterion."
  )
    .def(
      py::init<float>(),
      "value"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("value", &lincs::Performance::Real::get_value, "The numerical value of the real performance.")
    .def(py::pickle(
      [](const lincs::Performance::Real& real) {
        return py::make_tuple(real.get_value());
      },
      [](py::tuple t) {
        return lincs::Performance::Real(t[0].cast<float>());
      }
    ))
  ;

  py::class_<lincs::Performance::Integer>(
    performance_class,
    "Integer",
    "A performance for an integer-valued criterion."
  )
    .def(
      py::init<int>(),
      "value"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("value", &lincs::Performance::Integer::get_value, "The numerical value of the integer performance.")
    .def(py::pickle(
      [](const lincs::Performance::Integer& integer) {
        return py::make_tuple(integer.get_value());
      },
      [](py::tuple t) {
        return lincs::Performance::Integer(t[0].cast<int>());
      }
    ))
  ;

  py::class_<lincs::Performance::Enumerated>(
    performance_class,
    "Enumerated",
    "A performance for a criterion taking enumerated values."
  )
    .def(
      py::init<std::string>(),
      "value"_a,
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("value", &lincs::Performance::Enumerated::get_value, "The string value of the enumerated performance.")
    .def(py::pickle(
      [](const lincs::Performance::Enumerated& enumerated) {
        return py::make_tuple(enumerated.get_value());
      },
      [](py::tuple t) {
        return lincs::Performance::Enumerated(t[0].cast<std::string>());
      }
    ))
  ;

  performance_class
    .def(
      py::init<lincs::Performance::Real>(),
      "performance"_a,
      "Constructor for a real-valued performance."
    )
    .def(
      py::init<lincs::Performance::Integer>(),
      "performance"_a,
      "Constructor for an integer-valued performance."
    )
    .def(
      py::init<lincs::Performance::Enumerated>(),
      "performance"_a,
      "Constructor for an enumerated performance."
    )
  ;

  py::class_<lincs::Alternative>(
    m,
    "Alternative",
    "An alternative, with its performance on each criterion, maybe classified."
  )
    .def(
      py::init<std::string, std::vector<lincs::Performance>, std::optional<unsigned>>(),
      "name"_a, "profile"_a, "category_index"_a=std::optional<unsigned>(),
      "Parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("name", &lincs::Alternative::get_name, "The name of the alternative.")
    .def_property_readonly("profile", &lincs::Alternative::get_profile, "The performance profile of the alternative.")
    .def_property("category_index", &lincs::Alternative::get_category_index, &lincs::Alternative::set_category_index, "The index of the category of the alternative, if it is classified.")
    .def(py::pickle(
      [](const lincs::Alternative& alternative) {
        return py::make_tuple(alternative.get_name(), alternative.get_profile(), alternative.get_category_index());
      },
      [](py::tuple t) {
        return lincs::Alternative(
          t[0].cast<std::string>(),
          t[1].cast<std::vector<lincs::Performance>>(),
          t[2].cast<std::optional<unsigned>>()
        );
      }
    ))
  ;

  py::class_<lincs::Alternatives>(
    m,
    "Alternatives",
    "A set of alternatives, maybe classified."
  )
    .def(
      py::init<const lincs::Problem&, const std::vector<lincs::Alternative>&>(),
      "problem"_a, "alternatives"_a,
      "The :py:class:`Alternatives` being initialized must correspond to the given :py:class:`Problem`. Other parameters map exactly to attributes with identical names."
    )
    .def_property_readonly("alternatives", &lincs::Alternatives::get_alternatives, "The :py:class:`Alternative` objects in this set.")
    .def(
      "dump",
      &dump_alternatives,
      "problem"_a, "out"_a,
      "Dump the set of alternatives to the provided ``.write``-supporting file-like object, in CSV format."
    )
    .def_static(
      "load",
      &load_alternatives,
      "problem"_a, "in"_a,
      "Load a set of alternatives (classified or not) from the provided ``.read``-supporting file-like object, in CSV format."
    )
    .def(py::pickle(
      [](const lincs::Alternatives& alternatives) {
        return py::make_tuple(alternatives.get_alternatives());
      },
      [](py::tuple t) {
        return lincs::Alternatives(
          lincs::Internal(),
          t[0].cast<std::vector<lincs::Alternative>>()
        );
      }
    ))
  ;
}

void define_io_classes(py::module& m) {
  py::register_exception<lincs::DataValidationException>(m, "DataValidationException");

  define_problem_classes(m);
  define_model_classes(m);
  define_alternative_classes(m);
}

}  // namespace lincs
