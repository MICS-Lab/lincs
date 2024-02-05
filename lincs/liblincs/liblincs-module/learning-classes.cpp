// Copyright 2023-2024 Vincent Jacques

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

  auto learn_wbp_class = bp::class_<lincs::LearnMrsortByWeightsProfilesBreed>(
    "LearnMrsortByWeightsProfilesBreed",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
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
        bp::arg("observers")=std::vector<lincs::LearnMrsortByWeightsProfilesBreed::Observer*>{}
      ),
      "@todo(Documentation, v1.1) Add a docstring."
    )[
      bp::with_custodian_and_ward<1, 2,
      bp::with_custodian_and_ward<1, 3,
      bp::with_custodian_and_ward<1, 4,
      bp::with_custodian_and_ward<1, 5,
      bp::with_custodian_and_ward<1, 6,
      bp::with_custodian_and_ward<1, 7,
      bp::with_custodian_and_ward<1, 8
    >>>>>>>()])
    .def("perform", &lincs::LearnMrsortByWeightsProfilesBreed::perform, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;

  {
    bp::scope scope(learn_wbp_class);

    bp::class_<lincs::LearnMrsortByWeightsProfilesBreed::LearningData, boost::noncopyable>(
      "LearningData",
      "@todo(Documentation, v1.1) Add a docstring.",
      bp::no_init
    )
      .def(bp::init<const lincs::Problem&, const lincs::Alternatives&, unsigned, unsigned>(
        (bp::arg("self"), "problem", "learning_set", "models_count", "random_seed"),
        "@todo(Documentation, v1.1) Add a docstring."
      )[bp::with_custodian_and_ward<1, 2 /* No reference kept on 'learning_set' => no custodian_and_ward */>()])
      // About the problem and learning set:
      .add_property("criteria_count", &lincs::PreProcessedLearningSet::criteria_count, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("categories_count", &lincs::PreProcessedLearningSet::categories_count, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("boundaries_count", &lincs::PreProcessedLearningSet::boundaries_count, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("alternatives_count", &lincs::PreProcessedLearningSet::alternatives_count, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("values_counts", &lincs::PreProcessedLearningSet::values_counts, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("performance_ranks", &lincs::PreProcessedLearningSet::performance_ranks, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("assignments", &lincs::PreProcessedLearningSet::assignments, "@todo(Documentation, v1.1) Add a docstring.")
      // About WPB:
      .add_property("models_count", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::models_count, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("urbgs", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::urbgs, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("iteration_index", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::iteration_index, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("model_indexes", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::model_indexes, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("accuracies", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::accuracies, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("profile_ranks", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::profile_ranks, "@todo(Documentation, v1.1) Add a docstring.")
      .add_property("weights", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::weights, "@todo(Documentation, v1.1) Add a docstring.")
      .def("get_best_accuracy", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::get_best_accuracy, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
      .def("get_best_model", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::get_best_model, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
    ;

    struct ProfilesInitializationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy> {
      void initialize_profiles(const unsigned begin, const unsigned end) override { this->get_override("initialize_profiles")(begin, end); }
    };

    bp::class_<ProfilesInitializationStrategyWrap, boost::noncopyable>(
      "ProfilesInitializationStrategy",
      "@todo(Documentation, v1.1) Add a docstring."
    )
      .def(
        "initialize_profiles",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy::initialize_profiles),
        (bp::arg("self"), "model_indexes_begin", "model_indexes_end"),
        "@todo(Documentation, v1.1) Add a docstring."
      )
    ;

    struct WeightsOptimizationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy> {
      void optimize_weights() override { this->get_override("optimize_weights")(); }
    };

    bp::class_<WeightsOptimizationStrategyWrap, boost::noncopyable>(
      "WeightsOptimizationStrategy",
      "@todo(Documentation, v1.1) Add a docstring."
    )
      .def(
        "optimize_weights",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy::optimize_weights),
        (bp::arg("self")),
        "@todo(Documentation, v1.1) Add a docstring."
      )
    ;

    struct ProfilesImprovementStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy> {
      void improve_profiles() override { this->get_override("improve_profiles")(); }
    };

    bp::class_<ProfilesImprovementStrategyWrap, boost::noncopyable>(
      "ProfilesImprovementStrategy",
      "@todo(Documentation, v1.1) Add a docstring."
    )
      .def(
        "improve_profiles",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy::improve_profiles),
        (bp::arg("self")),
        "@todo(Documentation, v1.1) Add a docstring."
      )
    ;

    struct BreedingStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy> {
      void breed() override { this->get_override("breed")(); }
    };

    bp::class_<BreedingStrategyWrap, boost::noncopyable>(
      "BreedingStrategy",
      "@todo(Documentation, v1.1) Add a docstring."
    )
      .def(
        "breed",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy::breed),
        (bp::arg("self")),
        "@todo(Documentation, v1.1) Add a docstring."
      )
    ;

    struct TerminationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy> {
      bool terminate() override { return this->get_override("terminate")(); }
    };

    bp::class_<TerminationStrategyWrap, boost::noncopyable>(
      "TerminationStrategy",
      "@todo(Documentation, v1.1) Add a docstring."
    )
      .def(
        "terminate",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy::terminate),
        (bp::arg("self")),
        "@todo(Documentation, v1.1) Add a docstring."
      )
    ;

    struct ObserverWrap : lincs::LearnMrsortByWeightsProfilesBreed::Observer, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::Observer> {
      void after_iteration() override { this->get_override("after_iteration")(); }
      void before_return() override { this->get_override("before_return")(); }
    };

    bp::class_<ObserverWrap, boost::noncopyable>(
      "Observer",
      "@todo(Documentation, v1.1) Add a docstring."
    )
      .def(
        "after_iteration",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::Observer::after_iteration),
        (bp::arg("self")),
        "@todo(Documentation, v1.1) Add a docstring."
      )
      .def(
        "before_return",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::Observer::before_return),
        (bp::arg("self")),
        "@todo(Documentation, v1.1) Add a docstring."
      )
    ;
  }

  bp::class_<
    lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy>
  >(
    "InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>(
      (bp::arg("self"), "learning_data"),
      "@todo(Documentation, v1.1) Add a docstring."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "initialize_profiles",
      &lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles,
      (bp::arg("self"), "model_indexes_begin", "model_indexes_end"),
      "@todo(Documentation, v1.1) Add a docstring."
    )
  ;

  bp::class_<
    lincs::OptimizeWeightsUsingGlop,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy>
  >(
    "OptimizeWeightsUsingGlop",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>(
      (bp::arg("self"), "learning_data"),
      "@todo(Documentation, v1.1) Add a docstring."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "optimize_weights",
      &lincs::OptimizeWeightsUsingGlop::optimize_weights,
      (bp::arg("self")),
      "@todo(Documentation, v1.1) Add a docstring."
    )
  ;

  bp::class_<
    lincs::OptimizeWeightsUsingAlglib,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy>
  >(
    "OptimizeWeightsUsingAlglib",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>(
      (bp::arg("self"), "learning_data"),
      "@todo(Documentation, v1.1) Add a docstring."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "optimize_weights",
      &lincs::OptimizeWeightsUsingAlglib::optimize_weights,
      (bp::arg("self")),
      "@todo(Documentation, v1.1) Add a docstring."
    )
  ;

  bp::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnCpu,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy>
  >(
    "ImproveProfilesWithAccuracyHeuristicOnCpu",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>((bp::arg("self"), "learning_data"), "@todo(Documentation, v1.1) Add a docstring.")[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "improve_profiles",
      &lincs::ImproveProfilesWithAccuracyHeuristicOnCpu::improve_profiles,
      (bp::arg("self")),
      "@todo(Documentation, v1.1) Add a docstring."
    )
  ;

  #ifdef LINCS_HAS_NVCC
  bp::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnGpu,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy>,
    boost::noncopyable
  >(
    "ImproveProfilesWithAccuracyHeuristicOnGpu",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>(
      (bp::arg("self"), "learning_data"),
      "@todo(Documentation, v1.1) Add a docstring."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "improve_profiles",
      &lincs::ImproveProfilesWithAccuracyHeuristicOnGpu::improve_profiles,
      (bp::arg("self")),
      "@todo(Documentation, v1.1) Add a docstring."
    )
  ;
  #endif  // LINCS_HAS_NVCC

  bp::class_<lincs::ReinitializeLeastAccurate, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy>>(
    "ReinitializeLeastAccurate",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy&, unsigned>(
      (bp::arg("self"), "learning_data", "profiles_initialization_strategy", "count"),
      "@todo(Documentation, v1.1) Add a docstring."
    )[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3>>()])
    .def("breed", &lincs::ReinitializeLeastAccurate::breed, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;

  bp::class_<lincs::TerminateAtAccuracy, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAtAccuracy",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, unsigned>(
      (bp::arg("self"), "learning_data", "target_accuracy"),
      "@todo(Documentation, v1.1) Add a docstring."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def("terminate", &lincs::TerminateAtAccuracy::terminate, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;

  bp::class_<lincs::TerminateAfterIterations, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterIterations",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, unsigned>(
      (bp::arg("self"), "learning_data", "max_iterations_count"),
      "@todo(Documentation, v1.1) Add a docstring."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def("terminate", &lincs::TerminateAfterIterations::terminate, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;

  bp::class_<lincs::TerminateAfterIterationsWithoutProgress, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterIterationsWithoutProgress",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, unsigned>(
      (bp::arg("self"), "learning_data", "max_iterations_count"),
      "@todo(Documentation, v1.1) Add a docstring."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def("terminate", &lincs::TerminateAfterIterationsWithoutProgress::terminate, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;

  bp::class_<lincs::TerminateAfterSeconds, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterSeconds",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<float>((bp::arg("self"), "max_seconds"), "@todo(Documentation, v1.1) Add a docstring."))
    .def("terminate", &lincs::TerminateAfterSeconds::terminate, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;

  bp::class_<lincs::TerminateAfterSecondsWithoutProgress, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterSecondsWithoutProgress",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, float>((bp::arg("self"), "learning_data", "max_seconds"), "@todo(Documentation, v1.1) Add a docstring.")[bp::with_custodian_and_ward<1, 2>()])
    .def("terminate", &lincs::TerminateAfterSecondsWithoutProgress::terminate, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;

  bp::class_<lincs::TerminateWhenAny, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateWhenAny",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<std::vector<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy*>>((bp::arg("self"), "termination_strategies"), "@todo(Documentation, v1.1) Add a docstring.")[bp::with_custodian_and_ward<1, 2>()])
    .def("terminate", &lincs::TerminateWhenAny::terminate, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;


  bp::class_<lincs::LearnUcncsBySatByCoalitionsUsingMinisat, boost::noncopyable>(
    "LearnUcncsBySatByCoalitionsUsingMinisat",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<const lincs::Problem&, const lincs::Alternatives&>((bp::arg("self"), "problem", "learning_set"), "@todo(Documentation, v1.1) Add a docstring.")[bp::with_custodian_and_ward<1, 2 /* No reference kept on 'learning_set' => no custodian_and_ward */>()])
    .def("perform", &lincs::LearnUcncsBySatByCoalitionsUsingMinisat::perform, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;

  bp::class_<lincs::LearnUcncsBySatBySeparationUsingMinisat, boost::noncopyable>(
    "LearnUcncsBySatBySeparationUsingMinisat",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<const lincs::Problem&, const lincs::Alternatives&>((bp::arg("self"), "problem", "learning_set"), "@todo(Documentation, v1.1) Add a docstring.")[bp::with_custodian_and_ward<1, 2 /* No reference kept on 'learning_set' => no custodian_and_ward */>()])
    .def("perform", &lincs::LearnUcncsBySatBySeparationUsingMinisat::perform, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;

  bp::class_<lincs::LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat, boost::noncopyable>(
    "LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<const lincs::Problem&, const lincs::Alternatives&, unsigned, unsigned, unsigned>(
      (bp::arg("self"), "problem", "learning_set", bp::arg("nb_minimize_threads") = 0, bp::arg("timeout_fast_minimize") = 60, bp::arg("coef_minimize_time") = 2),
      "@todo(Documentation, v1.1) Add a docstring."
    )[bp::with_custodian_and_ward<1, 2 /* No reference kept on 'learning_set' => no custodian_and_ward */>()])
    .def("perform", &lincs::LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat::perform, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;

  bp::class_<lincs::LearnUcncsByMaxSatBySeparationUsingEvalmaxsat, boost::noncopyable>(
    "LearnUcncsByMaxSatBySeparationUsingEvalmaxsat",
    "@todo(Documentation, v1.1) Add a docstring.",
    bp::no_init
  )
    .def(bp::init<const lincs::Problem&, const lincs::Alternatives&, unsigned, unsigned, unsigned>(
      (bp::arg("self"), "problem", "learning_set", bp::arg("nb_minimize_threads") = 0, bp::arg("timeout_fast_minimize") = 60, bp::arg("coef_minimize_time") = 2),
      "@todo(Documentation, v1.1) Add a docstring."
    )[bp::with_custodian_and_ward<1, 2 /* No reference kept on 'learning_set' => no custodian_and_ward */>()])
    .def("perform", &lincs::LearnUcncsByMaxSatBySeparationUsingEvalmaxsat::perform, (bp::arg("self")), "@todo(Documentation, v1.1) Add a docstring.")
  ;
}

}  // namespace lincs
