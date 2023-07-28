// Copyright 2023 Vincent Jacques

#include "lincs.hpp"

#include <iostream>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>

#include "vendored/magic_enum.hpp"

#ifndef DOCTEST_CONFIG_DISABLE
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif
#include "vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


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

lincs::LearnMrsortByWeightsProfilesBreed::LearningData* make_learning_data(
  const lincs::Problem& problem,
  const lincs::Alternatives& learning_set,
  const unsigned models_count,
  const unsigned random_seed
) {
  return new lincs::LearnMrsortByWeightsProfilesBreed::LearningData(std::move(
    lincs::LearnMrsortByWeightsProfilesBreed::LearningData::make(problem, learning_set, models_count, random_seed)));
}

std::optional<unsigned> get_alternative_category_index(const lincs::Alternative& alt) {
  return alt.category_index;
}

void set_alternative_category_index(lincs::Alternative& alt, std::optional<unsigned> category_index) {
  alt.category_index = category_index;
}

}  // namespace

template <typename T>
auto auto_enum(const std::string& name) {
  auto e = bp::enum_<T>(name.c_str());
  for(T value : magic_enum::enum_values<T>()) {
    e.value(std::string(magic_enum::enum_name(value)).c_str(), value);
  }
  return e;
}

BOOST_PYTHON_MODULE(liblincs) {
  iterable_converter()
    .from_python<std::vector<float>>()
    .from_python<std::vector<unsigned>>()
    .from_python<std::vector<std::vector<unsigned>>>()
    .from_python<std::vector<lincs::Category>>()
    .from_python<std::vector<lincs::Criterion>>()
    .from_python<std::vector<lincs::Model::Boundary>>()
    .from_python<std::vector<lincs::SufficientCoalitions>>()
    .from_python<std::vector<lincs::Alternative>>()
    .from_python<std::vector<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy*>>()
    .from_python<std::vector<lincs::LearnMrsortByWeightsProfilesBreed::Observer*>>()
  ;

  std_optional_converter<float>::enroll();
  std_optional_converter<unsigned>::enroll();

  auto criterion_class = bp::class_<lincs::Criterion>("Criterion", bp::init<std::string, lincs::Criterion::ValueType, lincs::Criterion::CategoryCorrelation>())
    .def_readwrite("name", &lincs::Criterion::name)
    .def_readwrite("value_type", &lincs::Criterion::value_type)
    .def_readwrite("category_correlation", &lincs::Criterion::category_correlation)
  ;
  // Note that nested things are at global scope as well. This is not wanted, not used, but doesn't hurt
  // because 'liblincs' is only partially imported into module 'lincs' (see '__init__.py').
  criterion_class.attr("ValueType") = auto_enum<lincs::Criterion::ValueType>("ValueType");
  criterion_class.attr("CategoryCorrelation") = auto_enum<lincs::Criterion::CategoryCorrelation>("CategoryCorrelation");

  bp::class_<lincs::Category>("Category", bp::init<std::string>())
    .def_readwrite("name", &lincs::Category::name)
  ;

  bp::class_<std::vector<lincs::Category>>("categories_vector")
    .def(bp::vector_indexing_suite<std::vector<lincs::Category>>())
  ;
  bp::class_<std::vector<lincs::Criterion>>("criteria_vector")
    .def(bp::vector_indexing_suite<std::vector<lincs::Criterion>>())
  ;

  auto problem_class = bp::class_<lincs::Problem>("Problem", bp::init<std::vector<lincs::Criterion>, std::vector<lincs::Category>>())
    .def_readwrite("criteria", &lincs::Problem::criteria)
    .def_readwrite("categories", &lincs::Problem::categories)
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
  bp::def(
    "generate_classification_problem",
    &lincs::generate_classification_problem,
    (bp::arg("criteria_count"), "categories_count", "random_seed"),
    "Generate a problem with `criteria_count` criteria and `categories_count` categories."
  );

  // @todo Double-check why we need both an enum 'Kind' and two tag classes 'Weights' and 'Roots'; simplify or document
  bp::class_<lincs::SufficientCoalitions::Weights>("Weights", bp::no_init);
  bp::class_<lincs::SufficientCoalitions::Roots>("Roots", bp::no_init);
  auto sufficient_coalitions_class = bp::class_<lincs::SufficientCoalitions>("SufficientCoalitions", bp::no_init)
    .def(bp::init<lincs::SufficientCoalitions::Weights, std::vector<float>>())
    .def(bp::init<lincs::SufficientCoalitions::Roots, unsigned, std::vector<std::vector<unsigned>>>())
    .def_readwrite("kind", &lincs::SufficientCoalitions::kind)
    .def_readwrite("criterion_weights", &lincs::SufficientCoalitions::criterion_weights)
  ;
  sufficient_coalitions_class.attr("Kind") = auto_enum<lincs::SufficientCoalitions::Kind>("Kind");
  sufficient_coalitions_class.attr("weights") = lincs::SufficientCoalitions::weights;
  sufficient_coalitions_class.attr("roots") = lincs::SufficientCoalitions::roots;

  bp::class_<std::vector<float>>("floats_vector")
    .def(bp::vector_indexing_suite<std::vector<float>>())
  ;
  auto model_class = bp::class_<lincs::Model>("Model", bp::init<const lincs::Problem&, const std::vector<lincs::Model::Boundary>&>())
    .def_readwrite("boundaries", &lincs::Model::boundaries)
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
  model_class.attr("Boundary") = bp::class_<lincs::Model::Boundary>("Boundary", bp::init<std::vector<float>, lincs::SufficientCoalitions>())
    .def_readwrite("profile", &lincs::Model::Boundary::profile)
    .def_readwrite("sufficient_coalitions", &lincs::Model::Boundary::sufficient_coalitions)
  ;
  bp::class_<std::vector<lincs::Model::Boundary>>("boundaries_vector")
    .def(bp::vector_indexing_suite<std::vector<lincs::Model::Boundary>>())
  ;
  bp::def(
    "generate_mrsort_classification_model",
    &lincs::generate_mrsort_classification_model,
    (bp::arg("problem"), "random_seed", bp::arg("fixed_weights_sum")=std::optional<float>()),
    "Generate an MR-Sort model for the provided `problem`."
  );

  bp::class_<lincs::Alternative>(
    "Alternative",
    bp::init<std::string, std::vector<float>, std::optional<unsigned>>(
      (bp::arg("name"), "profile", (bp::arg("category")=std::optional<unsigned>()))
    )
  )
    .def_readwrite("name", &lincs::Alternative::name)
    .def_readwrite("profile", &lincs::Alternative::profile)
    .add_property("category_index", &get_alternative_category_index, &set_alternative_category_index)
  ;
  bp::class_<std::vector<lincs::Alternative>>("alternatives_vector")
    .def(bp::vector_indexing_suite<std::vector<lincs::Alternative>>())
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
  bp::def(
    "generate_classified_alternatives",
    &lincs::generate_classified_alternatives,
    (bp::arg("problem"), "model", "alternatives_count", "random_seed", bp::arg("max_imbalance")=std::optional<float>()),
    "Generate a set of `alternatives_count` pseudo-random alternatives for the provided `problem`, classified according to the provided `model`."
  );
  bp::def(
    "misclassify_alternatives",
    &lincs::misclassify_alternatives,
    (bp::arg("problem"), "alternatives", "count", "random_seed"),
    "Misclassify `count` alternatives from the provided `alternatives`."
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

  auto learn_wbp_class = bp::class_<lincs::LearnMrsortByWeightsProfilesBreed>("LearnMrsortByWeightsProfilesBreed", bp::no_init)
    .def(bp::init<
      lincs::LearnMrsortByWeightsProfilesBreed::LearningData&,
      lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy&
    >(
      (
        bp::arg("learning_data"),
        "profiles_initialization_strategy",
        "weights_optimization_strategy",
        "profiles_improvement_strategy",
        "breeding_strategy",
        "termination_strategy"
      )
    ))
    .def(bp::init<
      lincs::LearnMrsortByWeightsProfilesBreed::LearningData&,
      lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy&,
      std::vector<lincs::LearnMrsortByWeightsProfilesBreed::Observer*>
    >(
      (
        bp::arg("learning_data"),
        "profiles_initialization_strategy",
        "weights_optimization_strategy",
        "profiles_improvement_strategy",
        "breeding_strategy",
        "termination_strategy",
        "observers"
      )
    ))
    .def("perform", &lincs::LearnMrsortByWeightsProfilesBreed::perform)
  ;

  learn_wbp_class.attr("LearningData") = bp::class_<lincs::LearnMrsortByWeightsProfilesBreed::LearningData, boost::noncopyable>("LearningData", bp::no_init)
    .def("make", &make_learning_data, bp::return_value_policy<bp::manage_new_object>()).staticmethod("make")
    // @todo Expose all attributes to allow non-trivial Python strategies and observers
    .def("get_best_accuracy", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::get_best_accuracy)
  ;

  struct ProfilesInitializationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy> {
    void initialize_profiles(const unsigned begin, const unsigned end) override { this->get_override("initialize_profiles")(begin, end); }
  };

  learn_wbp_class.attr("ProfilesInitializationStrategy") = bp::class_<ProfilesInitializationStrategyWrap, boost::noncopyable>("ProfilesInitializationStrategy")
    .def("initialize_profiles", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy::initialize_profiles))
  ;

  bp::class_<
    lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy>
  >(
    "InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>()
  )
    .def("initialize_profiles", &lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles)
  ;

  struct WeightsOptimizationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy> {
    void optimize_weights() override { this->get_override("optimize_weights")(); }
  };

  learn_wbp_class.attr("WeightsOptimizationStrategy") = bp::class_<WeightsOptimizationStrategyWrap, boost::noncopyable>("WeightsOptimizationStrategy")
    .def("optimize_weights", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy::optimize_weights))
  ;

  bp::class_<
    lincs::OptimizeWeightsUsingGlop,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy>
  >(
    "OptimizeWeightsUsingGlop",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>()
  )
    .def("optimize_weights", &lincs::OptimizeWeightsUsingGlop::optimize_weights)
  ;

  bp::class_<
    lincs::OptimizeWeightsUsingAlglib,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy>
  >(
    "OptimizeWeightsUsingAlglib",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>()
  )
    .def("optimize_weights", &lincs::OptimizeWeightsUsingAlglib::optimize_weights)
  ;

  struct ProfilesImprovementStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy> {
    void improve_profiles() override { this->get_override("improve_profiles")(); }
  };

  learn_wbp_class.attr("ProfilesImprovementStrategy") = bp::class_<ProfilesImprovementStrategyWrap, boost::noncopyable>("ProfilesImprovementStrategy")
    .def("improve_profiles", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy::improve_profiles))
  ;

  bp::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnCpu,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy>
  >(
    "ImproveProfilesWithAccuracyHeuristicOnCpu",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>()
  )
    .def("improve_profiles", &lincs::ImproveProfilesWithAccuracyHeuristicOnCpu::improve_profiles)
  ;

  #ifdef LINCS_HAS_NVCC
  bp::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnGpu,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy>,
    boost::noncopyable
  >(
    "ImproveProfilesWithAccuracyHeuristicOnGpu",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>()
  )
    .def("improve_profiles", &lincs::ImproveProfilesWithAccuracyHeuristicOnGpu::improve_profiles)
  ;
  #endif  // LINCS_HAS_NVCC

  struct BreedingStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy> {
    void breed() override { this->get_override("breed")(); }
  };

  learn_wbp_class.attr("BreedingStrategy") = bp::class_<BreedingStrategyWrap, boost::noncopyable>("BreedingStrategy")
    .def("breed", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy::breed))
  ;

  bp::class_<lincs::ReinitializeLeastAccurate, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy>>(
    "ReinitializeLeastAccurate",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy&, unsigned>()
  )
    .def("breed", &lincs::ReinitializeLeastAccurate::breed)
  ;

  struct TerminationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy> {
    bool terminate() override { return this->get_override("terminate")(); }
  };

  learn_wbp_class.attr("TerminationStrategy") = bp::class_<TerminationStrategyWrap, boost::noncopyable>("TerminationStrategy")
    .def("terminate", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy::terminate))
  ;

  bp::class_<lincs::TerminateAtAccuracy, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAtAccuracy",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, unsigned>()
  )
    .def("terminate", &lincs::TerminateAtAccuracy::terminate)
  ;

  bp::class_<lincs::TerminateAfterIterations, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterIterations",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, unsigned>()
  )
    .def("terminate", &lincs::TerminateAfterIterations::terminate)
  ;

  bp::class_<lincs::TerminateAfterSeconds, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterSeconds",
    bp::init<float>()
  )
    .def("terminate", &lincs::TerminateAfterSeconds::terminate)
  ;

  bp::class_<lincs::TerminateWhenAny, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateWhenAny",
    bp::init<std::vector<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy*>>()
  )
    .def("terminate", &lincs::TerminateWhenAny::terminate)
  ;

  struct ObserverWrap : lincs::LearnMrsortByWeightsProfilesBreed::Observer, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::Observer> {
    void after_iteration() override { this->get_override("after_iteration")(); }
  };

  learn_wbp_class.attr("Observer") = bp::class_<ObserverWrap, boost::noncopyable>("Observer")
    .def("after_iteration", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::Observer::after_iteration))
  ;


  bp::class_<lincs::LearnUcncsBySatByCoalitionsUsingMinisat>("LearnUcncsBySatByCoalitionsUsingMinisat", bp::init<const lincs::Problem&, const lincs::Alternatives&>())
    .def("perform", &lincs::LearnUcncsBySatByCoalitionsUsingMinisat::perform)
  ;

  bp::class_<lincs::LearnUcncsBySatBySeparationUsingMinisat>("LearnUcncsBySatBySeparationUsingMinisat", bp::init<const lincs::Problem&, const lincs::Alternatives&>())
    .def("perform", &lincs::LearnUcncsBySatBySeparationUsingMinisat::perform)
  ;
}
