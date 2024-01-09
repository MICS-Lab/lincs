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

template <typename T>
auto auto_enum(const char* name, const char* docstring = nullptr) {
  auto e = bp::enum_<T>(name, docstring);
  for(T value : magic_enum::enum_values<T>()) {
    e.value(std::string(magic_enum::enum_name(value)).c_str(), value);
  }
  return e;
}

}  // namespace

namespace lincs {

void define_problem_classes() {
  auto criterion_class = bp::class_<lincs::Criterion>("Criterion", "A classification criterion, to be used in a classification ``Problem``", bp::no_init)
    .def(bp::init<std::string, lincs::Criterion::RealValues>((bp::arg("self"), "name", "values"), "Constructor for real-valued criterion"))
    .def(bp::init<std::string, lincs::Criterion::IntegerValues>((bp::arg("self"), "name", "values"), "Constructor for integer-valued criterion"))
    .def(bp::init<std::string, lincs::Criterion::EnumeratedValues>((bp::arg("self"), "name", "values"), "Constructor for criterion with enumerated values"))
    .add_property("name", bp::make_function(&lincs::Criterion::get_name, bp::return_value_policy<bp::return_by_value>()), "The name of the criterion")
    .add_property("value_type", &lincs::Criterion::get_value_type, "The type of values for this criterion")
    .add_property("is_real", &lincs::Criterion::is_real, "``True`` if the criterion is real-valued")
    .add_property("is_integer", &lincs::Criterion::is_integer, "``True`` if the criterion is integer-valued")
    .add_property("is_enumerated", &lincs::Criterion::is_enumerated, "``True`` if the criterion takes enumerated values")
    .add_property("real_values", bp::make_function(&lincs::Criterion::get_real_values, bp::return_value_policy<bp::return_by_value>()), "Descriptor of the real values allowed for this criterion, accessible if ``is_real``")
    .add_property("integer_values", bp::make_function(&lincs::Criterion::get_integer_values, bp::return_value_policy<bp::return_by_value>()), "Descriptor of the integer values allowed for this criterion, accessible if ``is_integer``")
    .add_property("enumerated_values", bp::make_function(&lincs::Criterion::get_enumerated_values, bp::return_value_policy<bp::return_by_value>()), "Descriptor of the enumerated values allowed for this criterion, accessible if ``is_enumerated``")
    .def(bp::self == bp::self)  // Private, undocumented, used only for our tests
  ;

  // @todo(Project management, later) Find a way to avoid attributes (like "ValueType" below) to also be present at top-level in liblincs.
  // (Not a big issue though, as liblincs is carefully partially imported by '__init__.py', but this would allow 'from liblincs import *' instead.)
  criterion_class.attr("ValueType") = auto_enum<lincs::Criterion::ValueType>("ValueType", "The different types of values for a criterion");

  criterion_class.attr("PreferenceDirection") = auto_enum<lincs::Criterion::PreferenceDirection>("PreferenceDirection")
    .value("isotone", lincs::Criterion::PreferenceDirection::isotone)
    .value("antitone", lincs::Criterion::PreferenceDirection::antitone)
  ;

  criterion_class.attr("RealValues") = bp::class_<lincs::Criterion::RealValues>(
    "RealValues",
    "Descriptor of the real values allowed for a criterion",
    bp::init<lincs::Criterion::PreferenceDirection, float, float>(
      (bp::arg("self"), "preference_direction", "min_value", "max_value")
    )
  )
    .add_property("preference_direction", &lincs::Criterion::RealValues::get_preference_direction, "The preference direction for this criterion")
    .add_property("is_increasing", &lincs::Criterion::RealValues::is_increasing, "``True`` if the criterion has increasing preference direction")
    .add_property("is_decreasing", &lincs::Criterion::RealValues::is_decreasing, "``True`` if the criterion has decreasing preference direction")
    .add_property("min_value", &lincs::Criterion::RealValues::get_min_value, "The minimum value allowed for this criterion")
    .add_property("max_value", &lincs::Criterion::RealValues::get_max_value, "The maximum value allowed for this criterion")
  ;
  criterion_class.attr("IntegerValues") = bp::class_<lincs::Criterion::IntegerValues>(
    "IntegerValues",
    "Descriptor of the integer values allowed for a criterion",
    bp::init<lincs::Criterion::PreferenceDirection, int, int>(
      (bp::arg("self"), "preference_direction", "min_value", "max_value")
    )
  )
    .add_property("preference_direction", &lincs::Criterion::IntegerValues::get_preference_direction, "The preference direction for this criterion")
    .add_property("is_increasing", &lincs::Criterion::IntegerValues::is_increasing, "``True`` if the criterion has increasing preference direction")
    .add_property("is_decreasing", &lincs::Criterion::IntegerValues::is_decreasing, "``True`` if the criterion has decreasing preference direction")
    .add_property("min_value", &lincs::Criterion::IntegerValues::get_min_value, "The minimum value allowed for this criterion")
    .add_property("max_value", &lincs::Criterion::IntegerValues::get_max_value, "The maximum value allowed for this criterion")
  ;
  criterion_class.attr("EnumeratedValues") = bp::class_<lincs::Criterion::EnumeratedValues>(
    "EnumeratedValues",
    "Descriptor of the enumerated values allowed for a criterion",
    bp::init<std::vector<std::string>>(
      (bp::arg("self"), "ordered_values")
    )
  )
    .add_property("ordered_values", bp::make_function(&lincs::Criterion::EnumeratedValues::get_ordered_values, bp::return_value_policy<bp::return_by_value>()), "The values for this criterion, from the worst to the best")
    .def("get_value_rank", &lincs::Criterion::EnumeratedValues::get_value_rank, (bp::arg("self"), "value"), "Get the rank of a given value")
  ;

  bp::class_<lincs::Category>("Category", "A category of a classification ``Problem``", bp::init<std::string>((bp::arg("self"), "name")))
    // @todo(Project management, v1.1) Investigate return policies and stop returning everything by values, where const refs would be more appropriate
    .add_property("name", bp::make_function(&lincs::Category::get_name, bp::return_value_policy<bp::return_by_value>()), "The name of this category")
  ;

  auto problem_class = bp::class_<lincs::Problem>("Problem", "A classification problem, with criteria and categories", bp::init<std::vector<lincs::Criterion>, std::vector<lincs::Category>>((bp::arg("self"), "criteria", "categories")))
    .add_property("criteria", bp::make_function(&lincs::Problem::get_criteria, bp::return_value_policy<bp::return_by_value>()), "The criteria of this problem")
    .add_property("ordered_categories", bp::make_function(&lincs::Problem::get_ordered_categories, bp::return_value_policy<bp::return_by_value>()), "The categories of this problem, from the worst to the best")
    .def(
      "dump",
      &dump_problem,
      (bp::arg("self"), "out"),
      "Dump the problem to the provided ``.write``-supporting file-like object, in YAML format"
    )
    .def(
      "load",
      &load_problem,
      (bp::arg("in")),
      "Load a problem from the provided ``.read``-supporting file-like object, in YAML format"
    )
    .staticmethod("load")
  ;
  problem_class.attr("JSON_SCHEMA") = lincs::Problem::json_schema;
}

void define_model_classes() {
  auto accepted_values_class = bp::class_<lincs::AcceptedValues>("AcceptedValues", bp::no_init)
    .def(bp::init<lincs::AcceptedValues::RealThresholds>((bp::args("self"), "values")))
    .def(bp::init<lincs::AcceptedValues::IntegerThresholds>((bp::args("self"), "values")))
    .def(bp::init<lincs::AcceptedValues::EnumeratedThresholds>((bp::args("self"), "values")))
    .add_property("value_type", &lincs::AcceptedValues::get_value_type, "The type of values for the corresponding criterion")
    .add_property("is_real", &lincs::AcceptedValues::is_real, "``True`` if the corresponding criterion is real-valued")
    .add_property("is_integer", &lincs::AcceptedValues::is_integer, "``True`` if the corresponding criterion is integer-valued")
    .add_property("is_enumerated", &lincs::AcceptedValues::is_enumerated, "``True`` if the corresponding criterion takes enumerated values")
    .add_property("kind", &lincs::AcceptedValues::get_kind, "The kind of descriptor for these accepted values")
    .add_property("is_thresholds", &lincs::AcceptedValues::is_thresholds, "``True`` if the descriptor is a set of thresholds")
    .add_property("real_thresholds", bp::make_function(&lincs::AcceptedValues::get_real_thresholds, bp::return_value_policy<bp::return_by_value>()), "Descriptor of the real thresholds, accessible if ``is_real and is_thresholds``")
    .add_property("integer_thresholds", bp::make_function(&lincs::AcceptedValues::get_integer_thresholds, bp::return_value_policy<bp::return_by_value>()), "Descriptor of the integer thresholds, accessible if ``is_integer and is_thresholds``")
    .add_property("enumerated_thresholds", bp::make_function(&lincs::AcceptedValues::get_enumerated_thresholds, bp::return_value_policy<bp::return_by_value>()), "Descriptor of the enumerated thresholds, accessible if ``is_enumerated and is_thresholds``")
  ;

  accepted_values_class.attr("Kind") = auto_enum<lincs::AcceptedValues::Kind>("Kind", "The different kinds of descriptors for accepted values");

  accepted_values_class.attr("RealThresholds") = bp::class_<lincs::AcceptedValues::RealThresholds>(
    "RealThresholds",
    "Descriptor for thresholds for an real-valued criterion",
    bp::init<const std::vector<float>&>((bp::arg("self"), "thresholds"))
  )
    .add_property("thresholds", bp::make_function(&lincs::AcceptedValues::RealThresholds::get_thresholds, bp::return_value_policy<bp::return_by_value>()), "The thresholds for this descriptor")
  ;

  accepted_values_class.attr("IntegerThresholds") = bp::class_<lincs::AcceptedValues::IntegerThresholds>(
    "IntegerThresholds",
    "Descriptor for thresholds for an integer-valued criterion",
    bp::init<const std::vector<int>&>((bp::arg("self"), "thresholds"))
  )
    .add_property("thresholds", bp::make_function(&lincs::AcceptedValues::IntegerThresholds::get_thresholds, bp::return_value_policy<bp::return_by_value>()), "The thresholds for this descriptor")
  ;

  accepted_values_class.attr("EnumeratedThresholds") = bp::class_<lincs::AcceptedValues::EnumeratedThresholds>(
    "EnumeratedThresholds",
    "Descriptor for thresholds for a criterion taking enumerated values",
    bp::init<const std::vector<std::string>&>((bp::arg("self"), "thresholds"))
  )
    .add_property("thresholds", bp::make_function(&lincs::AcceptedValues::EnumeratedThresholds::get_thresholds, bp::return_value_policy<bp::return_by_value>()), "The thresholds for this descriptor")
  ;

  auto sufficient_coalitions_class = bp::class_<lincs::SufficientCoalitions>("SufficientCoalitions", bp::no_init)
    .def(bp::init<lincs::SufficientCoalitions::Weights>((bp::args("self"), "weights")))
    .def(bp::init<lincs::SufficientCoalitions::Roots>((bp::args("self"), "roots")))
    .add_property("kind", &lincs::SufficientCoalitions::get_kind, "The kind of descriptor for these sufficient coalitions")
    .add_property("is_weights", &lincs::SufficientCoalitions::is_weights, "``True`` if the descriptor is a set of weights")
    .add_property("is_roots", &lincs::SufficientCoalitions::is_roots, "``True`` if the descriptor is a set of roots")
    .add_property("weights", bp::make_function(&lincs::SufficientCoalitions::get_weights, bp::return_value_policy<bp::return_by_value>()), "Descriptor of the weights, accessible if ``is_weights``")
    .add_property("roots", bp::make_function(&lincs::SufficientCoalitions::get_roots, bp::return_value_policy<bp::return_by_value>()), "Descriptor of the roots, accessible if ``is_roots``")
    .def(bp::self == bp::self)
  ;

  sufficient_coalitions_class.attr("Kind") = auto_enum<lincs::SufficientCoalitions::Kind>("Kind", "The different kinds of descriptors for sufficient coalitions");

  sufficient_coalitions_class.attr("Weights") = bp::class_<lincs::SufficientCoalitions::Weights>(
    "Weights",
    "Descriptor for sufficient coalitions defined by weights",
    bp::init<const std::vector<float>&>((bp::arg("self"), "criterion_weights"))
  )
    .add_property("criterion_weights", bp::make_function(&lincs::SufficientCoalitions::Weights::get_criterion_weights, bp::return_value_policy<bp::return_by_value>()), "The weights for each criterion")
  ;

  sufficient_coalitions_class.attr("Roots") = bp::class_<lincs::SufficientCoalitions::Roots>(
    "Roots",
    "Descriptor for sufficient coalitions defined by roots",
    bp::init<unsigned, const std::vector<std::vector<unsigned>>&>((bp::arg("self"), "criteria_count", "upset_roots"))
  )
    .add_property("upset_roots", bp::make_function(&lincs::SufficientCoalitions::Roots::get_upset_roots_as_vectors, bp::return_value_policy<bp::return_by_value>()), "The roots of the upset of sufficient coalitions")
  ;

  auto model_class = bp::class_<lincs::Model>(
    "Model",
    bp::init<const lincs::Problem&, const std::vector<lincs::AcceptedValues>&, const std::vector<lincs::SufficientCoalitions>&>((bp::arg("self"), "problem", "accepted_values", "sufficient_coalitions"))
  )
    .add_property("accepted_values", bp::make_function(&lincs::Model::get_accepted_values, bp::return_value_policy<bp::return_by_value>()), "The accepted values for each criterion")
    .add_property("sufficient_coalitions", bp::make_function(&lincs::Model::get_sufficient_coalitions, bp::return_value_policy<bp::return_by_value>()), "The sufficient coalitions for each category")
    .def(
      "dump",
      &dump_model,
      (bp::arg("self"), "problem", "out"),
      "Dump the model to the provided ``.write``-supporting file-like object, in YAML format"
    )
    .def(
      "load",
      &load_model,
      (bp::arg("problem"), "in"),
      // @todo(Documentation, v1.1) Search for all single back-quotes and double them where appropriate
      "Load a model for the provided `problem`, from the provided ``.read``-supporting file-like object, in YAML format"
    )
    .staticmethod("load")
  ;
  model_class.attr("JSON_SCHEMA") = lincs::Model::json_schema;
}

void define_alternative_classes() {
  auto performance_class = bp::class_<lincs::Performance>("Performance", bp::no_init)
    .def(bp::init<lincs::Performance::RealPerformance>((bp::arg("self"), "performance")))
    .def(bp::init<lincs::Performance::IntegerPerformance>((bp::arg("self"), "performance")))
    .def(bp::init<lincs::Performance::EnumeratedPerformance>((bp::arg("self"), "performance")))
    .add_property("value_type", &lincs::Performance::get_value_type)
    .add_property("is_real", &lincs::Performance::is_real)
    .add_property("is_integer", &lincs::Performance::is_integer)
    .add_property("is_enumerated", &lincs::Performance::is_enumerated)
    .add_property("real",  bp::make_function(&lincs::Performance::get_real, bp::return_value_policy<bp::return_by_value>()))
    .add_property("integer",  bp::make_function(&lincs::Performance::get_integer, bp::return_value_policy<bp::return_by_value>()))
    .add_property("enumerated",  bp::make_function(&lincs::Performance::get_enumerated, bp::return_value_policy<bp::return_by_value>()))
  ;

  performance_class.attr("RealPerformance") = bp::class_<lincs::Performance::RealPerformance>(
    "RealPerformance",
    bp::init<float>((bp::arg("self"), "value"))
  )
    .add_property("value", &lincs::Performance::RealPerformance::get_value)
  ;

  performance_class.attr("IntegerPerformance") = bp::class_<lincs::Performance::IntegerPerformance>(
    "IntegerPerformance",
    bp::init<int>((bp::arg("self"), "value"))
  )
    .add_property("value", &lincs::Performance::IntegerPerformance::get_value)
  ;

  performance_class.attr("EnumeratedPerformance") = bp::class_<lincs::Performance::EnumeratedPerformance>(
    "EnumeratedPerformance",
    bp::init<std::string>((bp::arg("self"), "value"))
  )
    .add_property("value", bp::make_function(&lincs::Performance::EnumeratedPerformance::get_value, bp::return_value_policy<bp::return_by_value>()))
  ;

  bp::class_<lincs::Alternative>(
    "Alternative",
    bp::init<std::string, std::vector<lincs::Performance>, std::optional<unsigned>>(
      (bp::arg("self"), "name", "profile", (bp::arg("category")=std::optional<unsigned>()))
    )
  )
    .add_property("name", bp::make_function(&lincs::Alternative::get_name, bp::return_value_policy<bp::return_by_value>()))
    .add_property("profile", bp::make_function(&lincs::Alternative::get_profile, bp::return_value_policy<bp::return_by_value>()))
    .add_property("category_index", bp::make_function(&lincs::Alternative::get_category_index, bp::return_value_policy<bp::return_by_value>()), &lincs::Alternative::set_category_index)
  ;
  bp::class_<lincs::Alternatives>(
    "Alternatives",
    bp::init<const lincs::Problem&, const std::vector<lincs::Alternative>&>((bp::arg("self"), "problem", "alternatives"))
  )
    .add_property("alternatives", bp::make_function(&lincs::Alternatives::get_alternatives, bp::return_value_policy<bp::return_by_value>()))
    .def(
      "dump",
      &dump_alternatives,
      (bp::arg("self"), "problem", "out"),
      "Dump the set of alternatives to the provided ``.write``-supporting file-like object, in CSV format."
    )
    .def(
      "load",
      &load_alternatives,
      (bp::arg("problem"), "in"),
      "Load a set of alternatives (classified or not) from the provided ``.read``-supporting file-like object, in CSV format."
    )
    .staticmethod("load")
  ;
}

void define_io_classes() {
  // @todo(Documentation, v1.1) Add a docstring to the exception class
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
