// Copyright 2023 Vincent Jacques

#include <Python.h>
// https://bugs.python.org/issue36020#msg371558
#undef snprintf
#undef vsnprintf

#include <boost/python.hpp>

#include "../learning.hpp"


namespace bp = boost::python;

namespace lincs {

void define_learning_classes() {
  PyObject* LearningFailureException_wrapper = PyErr_NewException("liblincs.LearningFailureException", PyExc_RuntimeError, NULL);

  bp::register_exception_translator<lincs::LearningFailureException>(
    [LearningFailureException_wrapper](const lincs::LearningFailureException& e) {
      PyErr_SetString(LearningFailureException_wrapper, e.what());
    }
  );

  bp::scope().attr("LearningFailureException") = bp::handle<>(bp::borrowed(LearningFailureException_wrapper));

  auto learn_wbp_class = bp::class_<lincs::LearnMrsortByWeightsProfilesBreed>("LearnMrsortByWeightsProfilesBreed", bp::no_init)
    // @todo(Project management, v1.1) Merge theses two constructors with a default empty list of observers
    .def(bp::init<
      lincs::LearnMrsortByWeightsProfilesBreed::LearningData&,
      lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy&,
      lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy&
    >(
      (
        bp::arg("self"),
        "learning_data",
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
        bp::arg("self"),
        "learning_data",
        "profiles_initialization_strategy",
        "weights_optimization_strategy",
        "profiles_improvement_strategy",
        "breeding_strategy",
        "termination_strategy",
        "observers"
      )
    ))
    .def("perform", &lincs::LearnMrsortByWeightsProfilesBreed::perform, (bp::arg("self")))
  ;

  learn_wbp_class.attr("LearningData") =
    bp::class_<lincs::LearnMrsortByWeightsProfilesBreed::LearningData, boost::noncopyable>(
      "LearningData",
      bp::init<const lincs::Problem&, const lincs::Alternatives&, unsigned, unsigned>((bp::arg("self"), "problem", "learning_set", "models_count", "random_seed"))
    )
    .def("get_best_accuracy", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::get_best_accuracy, (bp::arg("self")))
    .def_readonly("iteration_index", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::iteration_index)
  ;

  struct ProfilesInitializationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy> {
    void initialize_profiles(const unsigned begin, const unsigned end) override { this->get_override("initialize_profiles")(begin, end); }
  };

  learn_wbp_class.attr("ProfilesInitializationStrategy") = bp::class_<ProfilesInitializationStrategyWrap, boost::noncopyable>("ProfilesInitializationStrategy")
    .def("initialize_profiles", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy::initialize_profiles), (bp::arg("self"), "model_indexes_begin", "model_indexes_end"))
  ;

  bp::class_<
    lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy>
  >(
    "InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>((bp::arg("self"), "learning_data"))
  )
    .def("initialize_profiles", &lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles, (bp::arg("self"), "model_indexes_begin", "model_indexes_end"))
  ;

  struct WeightsOptimizationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy> {
    void optimize_weights() override { this->get_override("optimize_weights")(); }
  };

  learn_wbp_class.attr("WeightsOptimizationStrategy") = bp::class_<WeightsOptimizationStrategyWrap, boost::noncopyable>("WeightsOptimizationStrategy")
    .def("optimize_weights", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy::optimize_weights), (bp::arg("self")))
  ;

  bp::class_<
    lincs::OptimizeWeightsUsingGlop,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy>
  >(
    "OptimizeWeightsUsingGlop",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>((bp::arg("self"), "learning_data"))
  )
    .def("optimize_weights", &lincs::OptimizeWeightsUsingGlop::optimize_weights, (bp::arg("self")))
  ;

  bp::class_<
    lincs::OptimizeWeightsUsingAlglib,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy>
  >(
    "OptimizeWeightsUsingAlglib",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>((bp::arg("self"), "learning_data"))
  )
    .def("optimize_weights", &lincs::OptimizeWeightsUsingAlglib::optimize_weights, (bp::arg("self")))
  ;

  struct ProfilesImprovementStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy> {
    void improve_profiles() override { this->get_override("improve_profiles")(); }
  };

  learn_wbp_class.attr("ProfilesImprovementStrategy") = bp::class_<ProfilesImprovementStrategyWrap, boost::noncopyable>("ProfilesImprovementStrategy")
    .def("improve_profiles", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy::improve_profiles), (bp::arg("self")))
  ;

  bp::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnCpu,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy>
  >(
    "ImproveProfilesWithAccuracyHeuristicOnCpu",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>((bp::arg("self"), "learning_data"))
  )
    .def("improve_profiles", &lincs::ImproveProfilesWithAccuracyHeuristicOnCpu::improve_profiles, (bp::arg("self")))
  ;

  #ifdef LINCS_HAS_NVCC
  bp::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnGpu,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy>,
    boost::noncopyable
  >(
    "ImproveProfilesWithAccuracyHeuristicOnGpu",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>((bp::arg("self"), "learning_data"))
  )
    .def("improve_profiles", &lincs::ImproveProfilesWithAccuracyHeuristicOnGpu::improve_profiles, (bp::arg("self")))
  ;
  #endif  // LINCS_HAS_NVCC

  struct BreedingStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy> {
    void breed() override { this->get_override("breed")(); }
  };

  learn_wbp_class.attr("BreedingStrategy") = bp::class_<BreedingStrategyWrap, boost::noncopyable>("BreedingStrategy")
    .def("breed", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy::breed), (bp::arg("self")))
  ;

  bp::class_<lincs::ReinitializeLeastAccurate, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy>>(
    "ReinitializeLeastAccurate",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy&, unsigned>((bp::arg("self"), "learning_data", "profiles_initialization_strategy", "count"))
  )
    .def("breed", &lincs::ReinitializeLeastAccurate::breed, (bp::arg("self")))
  ;

  struct TerminationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy> {
    bool terminate() override { return this->get_override("terminate")(); }
  };

  learn_wbp_class.attr("TerminationStrategy") = bp::class_<TerminationStrategyWrap, boost::noncopyable>("TerminationStrategy")
    .def("terminate", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy::terminate), (bp::arg("self")))
  ;

  bp::class_<lincs::TerminateAtAccuracy, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAtAccuracy",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, unsigned>((bp::arg("self"), "learning_data", "target_accuracy"))
  )
    .def("terminate", &lincs::TerminateAtAccuracy::terminate, (bp::arg("self")))
  ;

  bp::class_<lincs::TerminateAfterIterations, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterIterations",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, unsigned>((bp::arg("self"), "learning_data", "max_iteration_index"))
  )
    .def("terminate", &lincs::TerminateAfterIterations::terminate, (bp::arg("self")))
  ;

  bp::class_<lincs::TerminateAfterIterationsWithoutProgress, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterIterationsWithoutProgress",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, unsigned>((bp::arg("self"), "learning_data", "max_iterations_count"))
  )
    .def("terminate", &lincs::TerminateAfterIterationsWithoutProgress::terminate, (bp::arg("self")))
  ;

  bp::class_<lincs::TerminateAfterSeconds, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterSeconds",
    bp::init<float>((bp::arg("self"), "max_seconds"))
  )
    .def("terminate", &lincs::TerminateAfterSeconds::terminate, (bp::arg("self")))
  ;

  bp::class_<lincs::TerminateAfterSecondsWithoutProgress, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterSecondsWithoutProgress",
    bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, float>((bp::arg("self"), "learning_data", "max_seconds"))
  )
    .def("terminate", &lincs::TerminateAfterSecondsWithoutProgress::terminate, (bp::arg("self")))
  ;

  bp::class_<lincs::TerminateWhenAny, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateWhenAny",
    bp::init<std::vector<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy*>>((bp::arg("self"), "termination_strategies"))
  )
    .def("terminate", &lincs::TerminateWhenAny::terminate, (bp::arg("self")))
  ;

  struct ObserverWrap : lincs::LearnMrsortByWeightsProfilesBreed::Observer, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::Observer> {
    void after_iteration() override { this->get_override("after_iteration")(); }
    void before_return() override { this->get_override("before_return")(); }
  };

  learn_wbp_class.attr("Observer") = bp::class_<ObserverWrap, boost::noncopyable>("Observer")
    .def("after_iteration", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::Observer::after_iteration), (bp::arg("self")))
    .def("before_return", bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::Observer::before_return), (bp::arg("self")))
  ;


  bp::class_<lincs::LearnUcncsBySatByCoalitionsUsingMinisat, boost::noncopyable>(
    "LearnUcncsBySatByCoalitionsUsingMinisat",
    bp::init<const lincs::Problem&, const lincs::Alternatives&>((bp::arg("self"), "problem", "learning_set"))
  )
    .def("perform", &lincs::LearnUcncsBySatByCoalitionsUsingMinisat::perform, (bp::arg("self")))
  ;

  bp::class_<lincs::LearnUcncsBySatBySeparationUsingMinisat, boost::noncopyable>(
    "LearnUcncsBySatBySeparationUsingMinisat",
    bp::init<const lincs::Problem&, const lincs::Alternatives&>((bp::arg("self"), "problem", "learning_set"))
  )
    .def("perform", &lincs::LearnUcncsBySatBySeparationUsingMinisat::perform, (bp::arg("self")))
  ;

  bp::class_<lincs::LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat, boost::noncopyable>(
    "LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat",
    bp::init<const lincs::Problem&, const lincs::Alternatives&>((bp::arg("self"), "problem", "learning_set"))
  )
    .def("perform", &lincs::LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat::perform, (bp::arg("self")))
  ;

  bp::class_<lincs::LearnUcncsByMaxSatBySeparationUsingEvalmaxsat, boost::noncopyable>(
    "LearnUcncsByMaxSatBySeparationUsingEvalmaxsat",
    bp::init<const lincs::Problem&, const lincs::Alternatives&>((bp::arg("self"), "problem", "learning_set"))
  )
    .def("perform", &lincs::LearnUcncsByMaxSatBySeparationUsingEvalmaxsat::perform, (bp::arg("self")))
  ;
}

}  // namespace lincs
