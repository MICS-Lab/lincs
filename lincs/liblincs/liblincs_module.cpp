// Copyright 2023 Vincent Jacques

#include "lincs.hpp"

#include <iostream>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>
#include <magic_enum.hpp>

#ifndef DOCTEST_CONFIG_DISABLE
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif
#include <doctest.h>  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


// @todo Consider using pybind11, which advertises itself as an evolved and simplified version of Boost.Python
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

void dump_model(const lincs::Model& model, bp::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  model.dump(out_stream);
}

void dump_alternatives(const lincs::Alternatives& alternatives, bp::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  alternatives.dump(out_stream);
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

// @todo Thoroughly review all conversions between Python and C++ types.
// - read Boost.Python doc in details and understand the contract
// - homogenize converters (some were copy-pasted from SO answers and even ChatGPT)
// - double-check if/when we need to increment reference counts on Python objects
// https://stackoverflow.com/a/15940413/905845
// https://misspent.wordpress.com/2009/09/27/how-to-write-boost-python-converters/
struct iterable_converter {
  template <typename Container>
  iterable_converter& from_python() {
    bp::converter::registry::push_back(
      &iterable_converter::convertible,
      &iterable_converter::construct<Container>,
      bp::type_id<Container>());

    return *this;
  }

  static void* convertible(PyObject* object) {
    return PyObject_GetIter(object) ? object : NULL;
  }

  template <typename Container>
  static void construct(
    PyObject* object,
    bp::converter::rvalue_from_python_stage1_data* data
  ) {
    bp::handle<> handle(bp::borrowed(object));

    typedef bp::converter::rvalue_from_python_storage<Container> storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    typedef bp::stl_input_iterator<typename Container::value_type> iterator;

    new (storage) Container(iterator(bp::object(handle)), iterator());
    data->convertible = storage;
  }
};

template<typename T>
struct std_optional_converter {
  static PyObject* convert(const std::optional<T>& value) {
    if (value) {
      return bp::incref(bp::object(*value).ptr());
    } else {
      return bp::incref(bp::object().ptr());
    }
  }

  static void* convertible(PyObject* obj) {
    if (obj == Py_None) {
      return new std::optional<T>();
    } else if (PyNumber_Check(obj) || PyUnicode_Check(obj)) {
      return new std::optional<T>(bp::extract<T>(obj));
    } else {
      return nullptr;
    }
  }

  static void construct(PyObject* obj, bp::converter::rvalue_from_python_stage1_data* data) {
    void* storage = reinterpret_cast<bp::converter::rvalue_from_python_storage<std::optional<T>>*>(data)->storage.bytes;
    new (storage) std::optional<T>(*reinterpret_cast<std::optional<T>*>(convertible(obj)));
    data->convertible = storage;
  }

  static void enroll() {
    bp::to_python_converter<std::optional<T>, std_optional_converter<T>>();
    bp::converter::registry::push_back(
      &std_optional_converter<T>::convertible,
      &std_optional_converter<T>::construct,
      bp::type_id<std::optional<T>>()
    );
  }
};

lincs::WeightsProfilesBreedMrSortLearning::Models* make_models(
  const lincs::Problem& problem,
  const lincs::Alternatives& learning_set,
  const unsigned models_count,
  const unsigned random_seed
) {
  return new lincs::WeightsProfilesBreedMrSortLearning::Models(std::move(
    lincs::WeightsProfilesBreedMrSortLearning::Models::make(problem, learning_set, models_count, random_seed)));
}

lincs::ImproveProfilesWithAccuracyHeuristicOnGpu::GpuModels* make_gpu_models(
  const lincs::WeightsProfilesBreedMrSortLearning::Models& models
) {
  return new lincs::ImproveProfilesWithAccuracyHeuristicOnGpu::GpuModels(std::move(
    lincs::ImproveProfilesWithAccuracyHeuristicOnGpu::GpuModels::make(models)));
}

std::optional<std::string> get_alternative_category(const lincs::Alternative& alt) {
  return alt.category;
}

void set_alternative_category(lincs::Alternative& alt, std::optional<std::string> category) {
  alt.category = category;
}

}  // namespace

template <typename T>
void auto_enum(const std::string& name) {
  auto e = bp::enum_<T>(name.c_str());
  for(T value : magic_enum::enum_values<T>()) {
    e.value(std::string(magic_enum::enum_name(value)).c_str(), value);
  }
}

BOOST_PYTHON_MODULE(liblincs) {
  iterable_converter()
    .from_python<std::vector<float>>()
    .from_python<std::vector<lincs::Problem::Category>>()
    .from_python<std::vector<lincs::Problem::Criterion>>()
    .from_python<std::vector<lincs::Model::Boundary>>()
    .from_python<std::vector<lincs::Model::SufficientCoalitions>>()
    .from_python<std::vector<lincs::Alternative>>()
  ;

  std_optional_converter<float>::enroll();
  std_optional_converter<std::string>::enroll();

  // @todo Decide wether we nest types or not, use the same nesting in Python and C++
  auto_enum<lincs::Problem::Criterion::ValueType>("ValueType");
  auto_enum<lincs::Problem::Criterion::CategoryCorrelation>("CategoryCorrelation");
  auto_enum<lincs::Model::SufficientCoalitions::Kind>("SufficientCoalitionsKind");

  bp::class_<lincs::Problem::Criterion>("Criterion", bp::init<std::string, lincs::Problem::Criterion::ValueType, lincs::Problem::Criterion::CategoryCorrelation>())
    .def_readwrite("name", &lincs::Problem::Criterion::name)
    .def_readwrite("value_type", &lincs::Problem::Criterion::value_type)
    .def_readwrite("category_correlation", &lincs::Problem::Criterion::category_correlation)
  ;

  bp::class_<lincs::Problem::Category>("Category", bp::init<std::string>())
    .def_readwrite("name", &lincs::Problem::Category::name)
  ;

  bp::class_<std::vector<lincs::Problem::Category>>("categories_vector")
    .def(bp::vector_indexing_suite<std::vector<lincs::Problem::Category>>())
  ;
  bp::class_<std::vector<lincs::Problem::Criterion>>("criteria_vector")
    .def(bp::vector_indexing_suite<std::vector<lincs::Problem::Criterion>>())
  ;

  bp::scope().attr("PROBLEM_JSON_SCHEMA") = lincs::Problem::json_schema;
  bp::class_<lincs::Problem>("Problem", bp::init<std::vector<lincs::Problem::Criterion>, std::vector<lincs::Problem::Category>>())
    .def_readwrite("criteria", &lincs::Problem::criteria)
    .def_readwrite("categories", &lincs::Problem::categories)
    .def(
      "dump",
      &dump_problem,
      (bp::arg("self"), "out"),
      "Dump the problem to the provided `.write()`-supporting file-like object, in YAML format."
    )
  ;
  // @todo Make these 'staticmethod's of Alternatives. Same for other load and generate functions.
  bp::def(
    "load_problem",
    &load_problem,
    (bp::arg("in")),
    "Load a problem from the provided `.read()`-supporting file-like object, in YAML format."
  );
  bp::def(
    "generate_problem",
    &lincs::generate_problem,
    (bp::arg("criteria_count"), "categories_count", "random_seed"),
    "Generate a problem with `criteria_count` criteria and `categories_count` categories."
  );

  bp::class_<lincs::Model::SufficientCoalitions>("SufficientCoalitions", bp::init<lincs::Model::SufficientCoalitions::Kind, std::vector<float>>())
    .def_readwrite("kind", &lincs::Model::SufficientCoalitions::kind)
    .def_readwrite("criterion_weights", &lincs::Model::SufficientCoalitions::criterion_weights)
  ;

  bp::class_<std::vector<float>>("floats_vector")
    .def(bp::vector_indexing_suite<std::vector<float>>())
  ;
  bp::class_<lincs::Model::Boundary>("Boundary", bp::init<std::vector<float>, lincs::Model::SufficientCoalitions>())
    .def_readwrite("profile", &lincs::Model::Boundary::profile)
    .def_readwrite("sufficient_coalitions", &lincs::Model::Boundary::sufficient_coalitions)
  ;
  bp::class_<std::vector<lincs::Model::Boundary>>("boundaries_vector")
    .def(bp::vector_indexing_suite<std::vector<lincs::Model::Boundary>>())
  ;
  bp::scope().attr("MODEL_JSON_SCHEMA") = lincs::Model::json_schema;
  bp::class_<lincs::Model>("Model", bp::init<const lincs::Problem&, const std::vector<lincs::Model::Boundary>&>())
    .def_readwrite("boundaries", &lincs::Model::boundaries)
    .def(
      "dump",
      &dump_model,
      (bp::arg("self"), "out"),
      "Dump the model to the provided `.write()`-supporting file-like object, in YAML format."
    )
  ;
  bp::def(
    "load_model",
    &load_model,
    (bp::arg("problem"), "in"),
    "Load a model for the provided `problem`, from the provided `.read()`-supporting file-like object, in YAML format."
  );
  bp::def(
    "generate_mrsort_model",
    &lincs::generate_mrsort_model,
    (bp::arg("problem"), "random_seed", bp::arg("fixed_weights_sum")=std::optional<float>()),
    "Generate an MR-Sort model for the provided `problem`."
  );

  bp::class_<lincs::Alternative>(
    "Alternative",
    bp::init<std::string, std::vector<float>, std::optional<std::string>>(
      (bp::arg("name"), "profile", (bp::arg("category")=std::optional<std::string>()))
    )
  )
    .def_readwrite("name", &lincs::Alternative::name)
    .def_readwrite("profile", &lincs::Alternative::profile)
    .add_property("category", &get_alternative_category, &set_alternative_category)
  ;
  bp::class_<std::vector<lincs::Alternative>>("alternatives_vector")
    .def(bp::vector_indexing_suite<std::vector<lincs::Alternative>>())
  ;
  bp::class_<lincs::Alternatives>("Alternatives", bp::init<const lincs::Problem&, const std::vector<lincs::Alternative>&>())
    .def_readwrite("alternatives", &lincs::Alternatives::alternatives)
    .def(
      "dump",
      &dump_alternatives,
      (bp::arg("self"), "out"),
      "Dump the set of alternatives to the provided `.write()`-supporting file-like object, in CSV format."
    )
  ;
  bp::def(
    "load_alternatives",
    &load_alternatives,
    (bp::arg("problem"), "in"),
    "Load a set of alternatives (classified or not) from the provided `.read()`-supporting file-like object, in CSV format."
  );
  bp::def(
    "generate_alternatives",
    &lincs::generate_alternatives,
    (bp::arg("problem"), "model", "alternatives_count", "random_seed", bp::arg("max_imbalance")=std::optional<float>()),
    "Generate a set of `alternatives_count` pseudo-random alternatives for the provided `problem`, classified according to the provided `model`."
  );

  bp::class_<lincs::ClassificationResult>("ClassificationResult", bp::no_init)
    .def_readonly("changed", &lincs::ClassificationResult::changed)
    .def_readonly("unchanged", &lincs::ClassificationResult::unchanged)
  ;
  bp::def(
    "classify_alternatives",
    &lincs::classify_alternatives,
    (bp::arg("problem"), "model", "alternatives"),
    "Classify the provided `alternatives` according to the provided `model`."
  );

  bp::class_<lincs::WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy, boost::noncopyable>("ProfilesInitializationStrategy", bp::no_init)
    .def("initialize_profiles", bp::pure_virtual(&lincs::WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy::initialize_profiles));

  bp::class_<
    lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion,
    bp::bases<lincs::WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy>
  >(
    "InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion",
    bp::init<lincs::WeightsProfilesBreedMrSortLearning::Models&>()
  )
    .def("initialize_profiles", &lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles);

  bp::class_<lincs::WeightsProfilesBreedMrSortLearning::WeightsOptimizationStrategy, boost::noncopyable>("WeightsOptimizationStrategy", bp::no_init)
    .def("optimize_weights", bp::pure_virtual(&lincs::WeightsProfilesBreedMrSortLearning::WeightsOptimizationStrategy::optimize_weights));

  bp::class_<
    lincs::OptimizeWeightsUsingGlop,
    bp::bases<lincs::WeightsProfilesBreedMrSortLearning::WeightsOptimizationStrategy>
  >(
    "OptimizeWeightsUsingGlop",
    bp::init<lincs::WeightsProfilesBreedMrSortLearning::Models&>()
  )
    .def("optimize_weights", &lincs::OptimizeWeightsUsingGlop::optimize_weights);

  bp::class_<
    lincs::OptimizeWeightsUsingAlglib,
    bp::bases<lincs::WeightsProfilesBreedMrSortLearning::WeightsOptimizationStrategy>
  >(
    "OptimizeWeightsUsingAlglib",
    bp::init<lincs::WeightsProfilesBreedMrSortLearning::Models&>()
  )
    .def("optimize_weights", &lincs::OptimizeWeightsUsingAlglib::optimize_weights);

  bp::class_<lincs::WeightsProfilesBreedMrSortLearning::ProfilesImprovementStrategy, boost::noncopyable>("ProfilesImprovementStrategy", bp::no_init)
    .def("improve_profiles", bp::pure_virtual(&lincs::WeightsProfilesBreedMrSortLearning::ProfilesImprovementStrategy::improve_profiles));

  bp::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnCpu,
    bp::bases<lincs::WeightsProfilesBreedMrSortLearning::ProfilesImprovementStrategy>
  >(
    "ImproveProfilesWithAccuracyHeuristicOnCpu",
    bp::init<lincs::WeightsProfilesBreedMrSortLearning::Models&>()
  )
    .def("improve_profiles", &lincs::ImproveProfilesWithAccuracyHeuristicOnCpu::improve_profiles);

  bp::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnGpu,
    bp::bases<lincs::WeightsProfilesBreedMrSortLearning::ProfilesImprovementStrategy>
  >(
    "ImproveProfilesWithAccuracyHeuristicOnGpu",
    bp::init<lincs::WeightsProfilesBreedMrSortLearning::Models&, lincs::ImproveProfilesWithAccuracyHeuristicOnGpu::GpuModels&>()
  )
    .def("improve_profiles", &lincs::ImproveProfilesWithAccuracyHeuristicOnGpu::improve_profiles);

  struct TerminationStrategyWrap : lincs::WeightsProfilesBreedMrSortLearning::TerminationStrategy, bp::wrapper<lincs::WeightsProfilesBreedMrSortLearning::TerminationStrategy> {
    bool terminate(unsigned iteration_index, unsigned best_accuracy) override {
      return this->get_override("terminate")(iteration_index, best_accuracy);
    }
  };

  bp::class_<TerminationStrategyWrap, boost::noncopyable>("TerminationStrategy")
    .def("terminate", bp::pure_virtual(&lincs::WeightsProfilesBreedMrSortLearning::TerminationStrategy::terminate));

  bp::class_<lincs::TerminateAtAccuracy, bp::bases<lincs::WeightsProfilesBreedMrSortLearning::TerminationStrategy>>("TerminateAtAccuracy", bp::init<unsigned>())
    .def("terminate", &lincs::TerminateAtAccuracy::terminate);

  bp::class_<lincs::WeightsProfilesBreedMrSortLearning::Models, boost::noncopyable>("Models", bp::no_init);
  bp::def("make_models", &make_models, bp::return_value_policy<bp::manage_new_object>());

  bp::class_<lincs::ImproveProfilesWithAccuracyHeuristicOnGpu::GpuModels, boost::noncopyable>("GpuModels", bp::no_init);
  bp::def("make_gpu_models", &make_gpu_models, bp::return_value_policy<bp::manage_new_object>());

  bp::class_<lincs::WeightsProfilesBreedMrSortLearning>(
    "WeightsProfilesBreedMrSortLearning",
    bp::init<
      lincs::WeightsProfilesBreedMrSortLearning::Models&,
      lincs::WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy&,
      lincs::WeightsProfilesBreedMrSortLearning::WeightsOptimizationStrategy&,
      lincs::WeightsProfilesBreedMrSortLearning::ProfilesImprovementStrategy&,
      lincs::WeightsProfilesBreedMrSortLearning::TerminationStrategy&
    >()
  )
    .def("perform", &lincs::WeightsProfilesBreedMrSortLearning::perform)
  ;
}
