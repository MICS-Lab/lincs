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
    .add_property("name", &lincs::Criterion::get_name)
    .add_property("value_type", &lincs::Criterion::get_value_type)
    .add_property("is_real", &lincs::Criterion::is_real)
    .add_property("is_integer", &lincs::Criterion::is_integer)
    .add_property("is_enumerated", &lincs::Criterion::is_enumerated)
    .add_property("preference_direction", &lincs::Criterion::get_preference_direction)
    .add_property("is_increasing", &lincs::Criterion::is_increasing)
    .add_property("is_decreasing", &lincs::Criterion::is_decreasing)
    .add_property("real_min_value", &lincs::Criterion::get_real_min_value)
    .add_property("real_max_value", &lincs::Criterion::get_real_max_value)
    .add_property("integer_min_value", &lincs::Criterion::get_integer_min_value)
    .add_property("integer_max_value", &lincs::Criterion::get_integer_max_value)
    .add_property("ordered_values", &lincs::Criterion::get_ordered_values)
    .def("get_value_rank", &lincs::Criterion::get_value_rank)
  ;
  criterion_class.attr("make_real") = &lincs::Criterion::make_real;
  criterion_class.attr("make_integer") = &lincs::Criterion::make_integer;
  criterion_class.attr("make_enumerated") = &lincs::Criterion::make_enumerated;

  criterion_class.attr("ValueType") = auto_enum<lincs::Criterion::ValueType>("ValueType");

  auto preference_direction_enum = auto_enum<lincs::Criterion::PreferenceDirection>("PreferenceDirection");
  preference_direction_enum.value("isotone", lincs::Criterion::PreferenceDirection::isotone);
  preference_direction_enum.value("antitone", lincs::Criterion::PreferenceDirection::antitone);
  criterion_class.attr("PreferenceDirection") = preference_direction_enum;

  bp::class_<lincs::Category>("Category", bp::init<std::string>())
    .def_readwrite("name", &lincs::Category::name)
  ;

  auto problem_class = bp::class_<lincs::Problem>("Problem", bp::init<std::vector<lincs::Criterion>, std::vector<lincs::Category>>())
    .def_readwrite("criteria", &lincs::Problem::criteria)
    .def_readwrite("ordered_categories", &lincs::Problem::ordered_categories)
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
    .add_property("real_thresholds", &lincs::AcceptedValues::get_real_thresholds)
    .add_property("integer_thresholds", &lincs::AcceptedValues::get_integer_thresholds)
    .add_property("enumerated_thresholds", &lincs::AcceptedValues::get_enumerated_thresholds)
  ;
  accepted_values_class.attr("make_real_thresholds") = &lincs::AcceptedValues::make_real_thresholds;
  accepted_values_class.attr("make_integer_thresholds") = &lincs::AcceptedValues::make_integer_thresholds;
  accepted_values_class.attr("make_enumerated_thresholds") = &lincs::AcceptedValues::make_enumerated_thresholds;

  auto sufficient_coalitions_class = bp::class_<lincs::SufficientCoalitions>("SufficientCoalitions", bp::no_init)
    .add_property("kind", &lincs::SufficientCoalitions::get_kind)
    .add_property("is_weights", &lincs::SufficientCoalitions::is_weights)
    .add_property("is_roots", &lincs::SufficientCoalitions::is_roots)
    .add_property("criterion_weights", &lincs::SufficientCoalitions::get_criterion_weights)
    .add_property("upset_roots", &lincs::SufficientCoalitions::get_upset_roots_as_vectors)
    .def(bp::self == bp::self)
  ;
  sufficient_coalitions_class.attr("make_weights") = &lincs::SufficientCoalitions::make_weights;
  sufficient_coalitions_class.attr("make_roots") = &lincs::SufficientCoalitions::make_roots_from_vectors;

  sufficient_coalitions_class.attr("Kind") = auto_enum<lincs::SufficientCoalitions::Kind>("Kind");

  auto model_class = bp::class_<lincs::Model>("Model", bp::init<const lincs::Problem&, const std::vector<lincs::AcceptedValues>&, const std::vector<lincs::SufficientCoalitions>&>())
    .def_readwrite("accepted_values", &lincs::Model::accepted_values)
    .def_readwrite("sufficient_coalitions", &lincs::Model::sufficient_coalitions)
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
    .def_readwrite("name", &lincs::Alternative::name)
    .def_readwrite("profile", &lincs::Alternative::profile)
    .add_property("category_index", &get_alternative_category_index, &set_alternative_category_index)
  ;
  bp::class_<lincs::Alternatives>("Alternatives", bp::init<const lincs::Problem&, const std::vector<lincs::Alternative>&>())
    .def_readwrite("alternatives", &lincs::Alternatives::alternatives)
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
