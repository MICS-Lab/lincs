// Copyright 2023 Vincent Jacques

#include <Python.h>
// https://bugs.python.org/issue36020#msg371558
#undef snprintf
#undef vsnprintf

#include <boost/python.hpp>
#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>

#include "../io.hpp"
#include "../vendored/magic_enum.hpp"


namespace bp = boost::python;

namespace {

class PythonOutputDevice : public boost::iostreams::sink {
  public:

  explicit PythonOutputDevice(bp::object out_file_) : out_file(out_file_) {}

  std::streamsize write(const char* s, std::streamsize n) {
    out_file.attr("write")(std::string(s, n));
    return n;
  }

  private:
  bp::object out_file;
};

void dump_problem(const lincs::Problem& problem, bp::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  problem.dump(out_stream);
}

void dump_model(const lincs::Model& model, const lincs::Problem& problem, bp::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  model.dump(problem, out_stream);
}

void dump_alternatives(const lincs::Alternatives& alternatives, const lincs::Problem& problem, bp::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  alternatives.dump(problem, out_stream);
}

class PythonInputDevice : public boost::iostreams::source {
  public:

  explicit PythonInputDevice(bp::object in_file_) : in_file(in_file_) {}

  std::streamsize read(char* s, std::streamsize n) {
    std::string str = bp::extract<std::string>(in_file.attr("read")(n));
    std::copy(str.begin(), str.end(), s);
    return str.size();
  }

  private:
  bp::object in_file;
};

lincs::Problem load_problem(bp::object& in_file) {
  boost::iostreams::stream<PythonInputDevice> in_stream(in_file);
  return lincs::Problem::load(in_stream);
}

lincs::Model load_model(const lincs::Problem& problem, bp::object& in_file) {
  boost::iostreams::stream<PythonInputDevice> in_stream(in_file);
  return lincs::Model::load(problem, in_stream);
}

lincs::Alternatives load_alternatives(const lincs::Problem& problem, bp::object& in_file) {
  boost::iostreams::stream<PythonInputDevice> in_stream(in_file);
  return lincs::Alternatives::load(problem, in_stream);
}

std::optional<unsigned> get_alternative_category_index(const lincs::Alternative& alt) {
  return alt.category_index;
}

void set_alternative_category_index(lincs::Alternative& alt, std::optional<unsigned> category_index) {
  alt.category_index = category_index;
}

template <typename T>
auto auto_enum(const std::string& name) {
  auto e = bp::enum_<T>(name.c_str());
  for(T value : magic_enum::enum_values<T>()) {
    e.value(std::string(magic_enum::enum_name(value)).c_str(), value);
  }
  return e;
}

}  // namespace

namespace lincs {

void define_problem_classes() {
  auto criterion_class = bp::class_<lincs::Criterion>("Criterion", bp::no_init)
    .def(bp::init<std::string, lincs::Criterion::RealValues>())
    .def(bp::init<std::string, lincs::Criterion::IntegerValues>())
    .def(bp::init<std::string, lincs::Criterion::EnumeratedValues>())
    .add_property("name", bp::make_function(&lincs::Criterion::get_name, bp::return_value_policy<bp::return_by_value>()))
    .add_property("value_type", &lincs::Criterion::get_value_type)
    .add_property("is_real", &lincs::Criterion::is_real)
    .add_property("is_integer", &lincs::Criterion::is_integer)
    .add_property("is_enumerated", &lincs::Criterion::is_enumerated)
    .add_property("real_values", bp::make_function(&lincs::Criterion::get_real_values, bp::return_value_policy<bp::return_by_value>()))
    .add_property("integer_values", bp::make_function(&lincs::Criterion::get_integer_values, bp::return_value_policy<bp::return_by_value>()))
    .add_property("enumerated_values", bp::make_function(&lincs::Criterion::get_enumerated_values, bp::return_value_policy<bp::return_by_value>()))
    .def(bp::self == bp::self)
  ;

  // @todo(Project management, later) Find a way to avoid attributes (like "ValueType" below) to also be present at top-level in liblincs.
  // (Not a big issue though, as liblincs is carefully partially imported by '__init__.py', but this would allow 'from liblincs import *' instead.)
  criterion_class.attr("ValueType") = auto_enum<lincs::Criterion::ValueType>("ValueType");

  auto preference_direction_enum = auto_enum<lincs::Criterion::PreferenceDirection>("PreferenceDirection");
  preference_direction_enum.value("isotone", lincs::Criterion::PreferenceDirection::isotone);
  preference_direction_enum.value("antitone", lincs::Criterion::PreferenceDirection::antitone);
  criterion_class.attr("PreferenceDirection") = preference_direction_enum;

  criterion_class.attr("RealValues") = bp::class_<lincs::Criterion::RealValues>(
    "RealValues",
    bp::init<lincs::Criterion::PreferenceDirection, float, float>(
      (bp::arg("preference_direction"), "min_value", "max_value")
    )
  )
    .add_property("preference_direction", &lincs::Criterion::RealValues::get_preference_direction)
    .add_property("is_increasing", &lincs::Criterion::RealValues::is_increasing)
    .add_property("is_decreasing", &lincs::Criterion::RealValues::is_decreasing)
    .add_property("min_value", &lincs::Criterion::RealValues::get_min_value)
    .add_property("max_value", &lincs::Criterion::RealValues::get_max_value)
  ;
  criterion_class.attr("IntegerValues") = bp::class_<lincs::Criterion::IntegerValues>(
    "IntegerValues",
    bp::init<lincs::Criterion::PreferenceDirection, int, int>(
      (bp::arg("preference_direction"), "min_value", "max_value")
    )
  )
    .add_property("preference_direction", &lincs::Criterion::IntegerValues::get_preference_direction)
    .add_property("is_increasing", &lincs::Criterion::IntegerValues::is_increasing)
    .add_property("is_decreasing", &lincs::Criterion::IntegerValues::is_decreasing)
    .add_property("min_value", &lincs::Criterion::IntegerValues::get_min_value)
    .add_property("max_value", &lincs::Criterion::IntegerValues::get_max_value)
  ;
  criterion_class.attr("EnumeratedValues") = bp::class_<lincs::Criterion::EnumeratedValues>(
    "EnumeratedValues",
    bp::init<std::vector<std::string>>(
      (bp::arg("ordered_values"))
    )
  )
    .add_property("ordered_values", bp::make_function(&lincs::Criterion::EnumeratedValues::get_ordered_values, bp::return_value_policy<bp::return_by_value>()))
    .def("get_value_rank", &lincs::Criterion::EnumeratedValues::get_value_rank)
  ;

  bp::class_<lincs::Category>("Category", bp::init<std::string>())
    // @todo(Project management, v1.1) Investigate return policies and stop returning everything by values, where const refs would be more appropriate
    .add_property("name", bp::make_function(&lincs::Category::get_name, bp::return_value_policy<bp::return_by_value>()))
  ;

  auto problem_class = bp::class_<lincs::Problem>("Problem", bp::init<std::vector<lincs::Criterion>, std::vector<lincs::Category>>())
    .add_property("criteria", bp::make_function(&lincs::Problem::get_criteria, bp::return_value_policy<bp::return_by_value>()))
    .add_property("ordered_categories", bp::make_function(&lincs::Problem::get_ordered_categories, bp::return_value_policy<bp::return_by_value>()))
    .def(
      "dump",
      &dump_problem,
      (bp::arg("self"), "out"),
      "Dump the problem to the provided `.write()`-supporting file-like object, in YAML format."
    )
    .def(
      "load",
      &load_problem,
      (bp::arg("in")),
      "Load a problem from the provided `.read()`-supporting file-like object, in YAML format."
    )
    .staticmethod("load")
  ;
  problem_class.attr("JSON_SCHEMA") = lincs::Problem::json_schema;
}

void define_model_classes() {
  auto accepted_values_class = bp::class_<lincs::AcceptedValues>("AcceptedValues", bp::no_init)
    .def(bp::init<lincs::AcceptedValues::RealThresholds>())
    .def(bp::init<lincs::AcceptedValues::IntegerThresholds>())
    .def(bp::init<lincs::AcceptedValues::EnumeratedThresholds>())
    .add_property("value_type", &lincs::AcceptedValues::get_value_type)
    .add_property("is_real", &lincs::AcceptedValues::is_real)
    .add_property("is_integer", &lincs::AcceptedValues::is_integer)
    .add_property("is_enumerated", &lincs::AcceptedValues::is_enumerated)
    .add_property("kind", &lincs::AcceptedValues::get_kind)
    .add_property("is_thresholds", &lincs::AcceptedValues::is_thresholds)
    .add_property("real_thresholds", bp::make_function(&lincs::AcceptedValues::get_real_thresholds, bp::return_value_policy<bp::return_by_value>()))
    .add_property("integer_thresholds", bp::make_function(&lincs::AcceptedValues::get_integer_thresholds, bp::return_value_policy<bp::return_by_value>()))
    .add_property("enumerated_thresholds", bp::make_function(&lincs::AcceptedValues::get_enumerated_thresholds, bp::return_value_policy<bp::return_by_value>()))
  ;

  accepted_values_class.attr("Kind") = auto_enum<lincs::AcceptedValues::Kind>("Kind");

  accepted_values_class.attr("RealThresholds") = bp::class_<lincs::AcceptedValues::RealThresholds>(
    "RealThresholds",
    bp::init<const std::vector<float>&>((bp::arg("thresholds")))
  )
    .add_property("thresholds", bp::make_function(&lincs::AcceptedValues::RealThresholds::get_thresholds, bp::return_value_policy<bp::return_by_value>()))
  ;

  accepted_values_class.attr("IntegerThresholds") = bp::class_<lincs::AcceptedValues::IntegerThresholds>(
    "IntegerThresholds",
    bp::init<const std::vector<int>&>((bp::arg("thresholds")))
  )
    .add_property("thresholds", bp::make_function(&lincs::AcceptedValues::IntegerThresholds::get_thresholds, bp::return_value_policy<bp::return_by_value>()))
  ;

  accepted_values_class.attr("EnumeratedThresholds") = bp::class_<lincs::AcceptedValues::EnumeratedThresholds>(
    "EnumeratedThresholds",
    bp::init<const std::vector<std::string>&>((bp::arg("thresholds")))
  )
    .add_property("thresholds", bp::make_function(&lincs::AcceptedValues::EnumeratedThresholds::get_thresholds, bp::return_value_policy<bp::return_by_value>()))
  ;

  auto sufficient_coalitions_class = bp::class_<lincs::SufficientCoalitions>("SufficientCoalitions", bp::no_init)
    .def(bp::init<lincs::SufficientCoalitions::Weights>())
    .def(bp::init<lincs::SufficientCoalitions::Roots>())
    .add_property("kind", &lincs::SufficientCoalitions::get_kind)
    .add_property("is_weights", &lincs::SufficientCoalitions::is_weights)
    .add_property("is_roots", &lincs::SufficientCoalitions::is_roots)
    .add_property("weights", bp::make_function(&lincs::SufficientCoalitions::get_weights, bp::return_value_policy<bp::return_by_value>()))
    .add_property("roots", bp::make_function(&lincs::SufficientCoalitions::get_roots, bp::return_value_policy<bp::return_by_value>()))
    .def(bp::self == bp::self)
  ;

  sufficient_coalitions_class.attr("Kind") = auto_enum<lincs::SufficientCoalitions::Kind>("Kind");

  sufficient_coalitions_class.attr("Weights") = bp::class_<lincs::SufficientCoalitions::Weights>(
    "Weights",
    bp::init<const std::vector<float>&>((bp::arg("criterion_weights")))
  )
    .add_property("criterion_weights", bp::make_function(&lincs::SufficientCoalitions::Weights::get_criterion_weights, bp::return_value_policy<bp::return_by_value>()))
  ;

  sufficient_coalitions_class.attr("Roots") = bp::class_<lincs::SufficientCoalitions::Roots>(
    "Roots",
    bp::init<unsigned, const std::vector<std::vector<unsigned>>&>((bp::arg("criteria_count"), "upset_roots"))
  )
    .add_property("upset_roots", bp::make_function(&lincs::SufficientCoalitions::Roots::get_upset_roots_as_vectors, bp::return_value_policy<bp::return_by_value>()))
  ;

  auto model_class = bp::class_<lincs::Model>("Model", bp::init<const lincs::Problem&, const std::vector<lincs::AcceptedValues>&, const std::vector<lincs::SufficientCoalitions>&>())
    .add_property("accepted_values", bp::make_function(&lincs::Model::get_accepted_values, bp::return_value_policy<bp::return_by_value>()))
    .add_property("sufficient_coalitions", bp::make_function(&lincs::Model::get_sufficient_coalitions, bp::return_value_policy<bp::return_by_value>()))
    .def(
      "dump",
      &dump_model,
      (bp::arg("self"), "problem", "out"),
      "Dump the model to the provided `.write()`-supporting file-like object, in YAML format."
    )
    .def(
      "load",
      &load_model,
      (bp::arg("problem"), "in"),
      "Load a model for the provided `problem`, from the provided `.read()`-supporting file-like object, in YAML format."
    )
    .staticmethod("load")
  ;
  model_class.attr("JSON_SCHEMA") = lincs::Model::json_schema;
}

void define_alternative_classes() {
  bp::class_<lincs::Performance>("Performance", bp::no_init)
    .def("make_real", &lincs::Performance::make_real)
    .def("make_integer", &lincs::Performance::make_integer)
    .def("make_enumerated", &lincs::Performance::make_enumerated)
    .add_property("real_value", &lincs::Performance::get_real_value)
    .add_property("integer_value", &lincs::Performance::get_integer_value)
    .add_property("enumerated_value", &lincs::Performance::get_enumerated_value)
  ;

  bp::class_<lincs::Alternative>(
    "Alternative",
    bp::init<std::string, std::vector<lincs::Performance>, std::optional<unsigned>>(
      (bp::arg("name"), "profile", (bp::arg("category")=std::optional<unsigned>()))
    )
  )
    .def_readonly("name", &lincs::Alternative::name)
    .def_readonly("profile", &lincs::Alternative::profile)
    .add_property("category_index", &get_alternative_category_index, &set_alternative_category_index)
  ;
  bp::class_<lincs::Alternatives>("Alternatives", bp::init<const lincs::Problem&, const std::vector<lincs::Alternative>&>())
    .def_readonly("alternatives", &lincs::Alternatives::alternatives)
    .def(
      "dump",
      &dump_alternatives,
      (bp::arg("self"), "problem", "out"),
      "Dump the set of alternatives to the provided `.write()`-supporting file-like object, in CSV format."
    )
    .def(
      "load",
      &load_alternatives,
      (bp::arg("problem"), "in"),
      "Load a set of alternatives (classified or not) from the provided `.read()`-supporting file-like object, in CSV format."
    )
    .staticmethod("load")
  ;
}

void define_io_classes() {
  PyObject* DataValidationException_wrapper = PyErr_NewException("liblincs.DataValidationException", PyExc_RuntimeError, NULL);
  bp::register_exception_translator<lincs::DataValidationException>(
    [DataValidationException_wrapper](const lincs::DataValidationException& e) {
      PyErr_SetString(DataValidationException_wrapper, e.what());
    }
  );
  bp::scope().attr("DataValidationException") = bp::handle<>(bp::borrowed(DataValidationException_wrapper));

  define_problem_classes();
  define_model_classes();
  define_alternative_classes();
}

}  // namespace lincs
