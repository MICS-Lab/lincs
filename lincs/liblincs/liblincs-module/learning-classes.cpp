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
    "The approach described in Olivier Sobrie's PhD thesis to learn MR-Sort models.",
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
      "Constructor accepting the strategies to use for each step of the learning."
    )[
      bp::with_custodian_and_ward<1, 2,
      bp::with_custodian_and_ward<1, 3,
      bp::with_custodian_and_ward<1, 4,
      bp::with_custodian_and_ward<1, 5,
      bp::with_custodian_and_ward<1, 6,
      bp::with_custodian_and_ward<1, 7,
      bp::with_custodian_and_ward<1, 8
    >>>>>>>()])
    .def("perform", &lincs::LearnMrsortByWeightsProfilesBreed::perform, (bp::arg("self")), "Actually perform the learning and return the learned model.")
  ;

  {
    bp::scope scope(learn_wbp_class);

    bp::class_<lincs::LearnMrsortByWeightsProfilesBreed::LearningData, boost::noncopyable>(
      "LearningData",
      "Data shared by all the strategies used in this learning.",
      bp::no_init
    )
      .def(bp::init<const lincs::Problem&, const lincs::Alternatives&, unsigned, unsigned>(
        (bp::arg("self"), "problem", "learning_set", "models_count", "random_seed"),
        "Constructor, pre-processing the learning set into a simpler form for strategies."
      )[bp::with_custodian_and_ward<1, 2 /* No reference kept on 'learning_set' => no custodian_and_ward */>()])
      // About the problem and learning set:
      .add_property("criteria_count", &lincs::PreProcessedLearningSet::criteria_count, "Number of criteria in the :py:class:`Problem`.")
      .add_property("categories_count", &lincs::PreProcessedLearningSet::categories_count, "Number of categories in the :py:class:`Problem`.")
      .add_property("boundaries_count", &lincs::PreProcessedLearningSet::boundaries_count, "Number of boundaries in the :py:class:`Problem`, *i.e* ``categories_count - 1``.")
      .add_property("alternatives_count", &lincs::PreProcessedLearningSet::alternatives_count, "Number of alternatives in the ``learning_set``.")
      .add_property("values_counts", &lincs::PreProcessedLearningSet::values_counts, "Indexed by ``[criterion_index]``. Number of different values for each criterion, in the ``learning_set`` and min and max values for numerical criteria.")
      .add_property("performance_ranks", &lincs::PreProcessedLearningSet::performance_ranks, "Indexed by ``[criterion_index][alternative_index]``. Rank of each alternative in the ``learning_set`` for each criterion.")
      .add_property("assignments", &lincs::PreProcessedLearningSet::assignments, "Indexed by ``[alternative_index]``. Category index of each alternative in the ``learning_set``.")
      // About WPB:
      .add_property("models_count", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::models_count, "The number of in-progress models for this learning.")
      .add_property("urbgs", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::urbgs, "Indexed by ``[model_index]``. Random number generators associated to each in-progress model.")
      .add_property("iteration_index", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::iteration_index, "The index of the current iteration of the WPB algorithm.")
      .add_property("model_indexes", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::model_indexes, "Indexed by ``0`` to ``models_count - 1``. Indexes of in-progress models ordered by increasing accuracy.")
      .add_property("accuracies", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::accuracies, "Indexed by ``[model_index]``. Accuracy of each in-progress model.")
      .add_property("profile_ranks", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::profile_ranks, "Indexed by ``[model_index][profile_index][criterion_index]``. The current rank of each profile, for each model and criterion.")
      .add_property("weights", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::weights, "Indexed by ``[model_index][criterion_index]``. The current MR-Sort weight of each criterion for each model.")
      .def("get_best_accuracy", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::get_best_accuracy, (bp::arg("self")), "Return the accuracy of the best model so far.")
      .def("get_best_model", &lincs::LearnMrsortByWeightsProfilesBreed::LearningData::get_best_model, (bp::arg("self")), "Return the best model so far.")
    ;

    struct ProfilesInitializationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy> {
      void initialize_profiles(const unsigned begin, const unsigned end) override { this->get_override("initialize_profiles")(begin, end); }
    };

    bp::class_<ProfilesInitializationStrategyWrap, boost::noncopyable>(
      "ProfilesInitializationStrategy",
      "Abstract base class for profiles initialization strategies."
    )
      .def(
        "initialize_profiles",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy::initialize_profiles),
        (bp::arg("self"), "model_indexes_begin", "model_indexes_end"),
        "Method to override. Should initialize all ``profile_ranks`` of models at indexes in ``[model_indexes[i] for i in range(model_indexes_begin, model_indexes_end)]``."
      )
    ;

    struct WeightsOptimizationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy> {
      void optimize_weights(const unsigned begin, const unsigned end) override { this->get_override("optimize_weights")(begin, end); }
    };

    bp::class_<WeightsOptimizationStrategyWrap, boost::noncopyable>(
      "WeightsOptimizationStrategy",
      "Abstract base class for weights optimization strategies."
    )
      .def(
        "optimize_weights",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy::optimize_weights),
        (bp::arg("self"), "model_indexes_begin", "model_indexes_end"),
        "Method to override. Should optimize ``weights`` of models at indexes in ``[model_indexes[i] for i in range(model_indexes_begin, model_indexes_end)]``."
      )
    ;

    struct ProfilesImprovementStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy> {
      void improve_profiles(const unsigned begin, const unsigned end) override { this->get_override("improve_profiles")(begin, end); }
    };

    bp::class_<ProfilesImprovementStrategyWrap, boost::noncopyable>(
      "ProfilesImprovementStrategy",
      "Abstract base class for profiles improvement strategies."
    )
      .def(
        "improve_profiles",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy::improve_profiles),
        (bp::arg("self"), "model_indexes_begin", "model_indexes_end"),
        "Method to override. Should improve ``profile_ranks`` of models at indexes in ``[model_indexes[i] for i in range(model_indexes_begin, model_indexes_end)]``."
      )
    ;

    struct BreedingStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy> {
      void breed() override { this->get_override("breed")(); }
    };

    bp::class_<BreedingStrategyWrap, boost::noncopyable>(
      "BreedingStrategy",
      "Abstract base class for breeding strategies."
    )
      .def(
        "breed",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy::breed),
        (bp::arg("self")),
        "Method to override."
      )
    ;

    struct TerminationStrategyWrap : lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy> {
      bool terminate() override { return this->get_override("terminate")(); }
    };

    bp::class_<TerminationStrategyWrap, boost::noncopyable>(
      "TerminationStrategy",
      "Abstract base class for termination strategies."
    )
      .def(
        "terminate",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy::terminate),
        (bp::arg("self")),
        "Method to override. Should return ``True`` if the learning should stop, ``False`` otherwise."
      )
    ;

    struct ObserverWrap : lincs::LearnMrsortByWeightsProfilesBreed::Observer, bp::wrapper<lincs::LearnMrsortByWeightsProfilesBreed::Observer> {
      void after_iteration() override { this->get_override("after_iteration")(); }
      void before_return() override { this->get_override("before_return")(); }
    };

    bp::class_<ObserverWrap, boost::noncopyable>(
      "Observer",
      "Abstract base class for observation strategies."
    )
      .def(
        "after_iteration",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::Observer::after_iteration),
        (bp::arg("self")),
        "Method to override. Called after each iteration. Should not change anything in the learning data."
      )
      .def(
        "before_return",
        bp::pure_virtual(&lincs::LearnMrsortByWeightsProfilesBreed::Observer::before_return),
        (bp::arg("self")),
        "Method to override. Called just before returning the learned model. Should not change anything in the learning data."
      )
    ;
  }

  bp::class_<
    lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy>
  >(
    "InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion",
    "The profiles initialization strategy described in Olivier Sobrie's PhD thesis.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>(
      (bp::arg("self"), "learning_data"),
      "Constructor. Keeps a reference to the learning data."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "initialize_profiles",
      &lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles,
      (bp::arg("self"), "model_indexes_begin", "model_indexes_end"),
      "Overrides the base method."
    )
  ;

  bp::class_<
    lincs::OptimizeWeightsUsingGlop,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy>
  >(
    "OptimizeWeightsUsingGlop",
    "The weights optimization strategy described in Olivier Sobrie's PhD thesis. The linear program is solved using GLOP.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>(
      (bp::arg("self"), "learning_data"),
      "Constructor. Keeps a reference to the learning data."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "optimize_weights",
      &lincs::OptimizeWeightsUsingGlop::optimize_weights,
      (bp::arg("self"), "model_indexes_begin", "model_indexes_end"),
      "Overrides the base method."
    )
  ;

  bp::class_<
    lincs::OptimizeWeightsUsingAlglib,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy>
  >(
    "OptimizeWeightsUsingAlglib",
    "The weights optimization strategy described in Olivier Sobrie's PhD thesis. The linear program is solved using AlgLib.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>(
      (bp::arg("self"), "learning_data"),
      "Constructor. Keeps a reference to the learning data."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "optimize_weights",
      &lincs::OptimizeWeightsUsingAlglib::optimize_weights,
      (bp::arg("self"), "model_indexes_begin", "model_indexes_end"),
      "Overrides the base method."
    )
  ;

  bp::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnCpu,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy>
  >(
    "ImproveProfilesWithAccuracyHeuristicOnCpu",
    "The profiles improvement strategy described in Olivier Sobrie's PhD thesis. Run on the CPU.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>(
      (bp::arg("self"), "learning_data"),
      "Constructor. Keeps a reference to the learning data."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "improve_profiles",
      &lincs::ImproveProfilesWithAccuracyHeuristicOnCpu::improve_profiles,
      (bp::arg("self"), "model_indexes_begin", "model_indexes_end"),
      "Overrides the base method."
    )
  ;

  #ifdef LINCS_HAS_NVCC
  bp::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnGpu,
    bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy>,
    boost::noncopyable
  >(
    "ImproveProfilesWithAccuracyHeuristicOnGpu",
    "The profiles improvement strategy described in Olivier Sobrie's PhD thesis. Run on the CUDA-capable GPU.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&>(
      (bp::arg("self"), "learning_data"),
      "Constructor. Keeps a reference to the learning data."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "improve_profiles",
      &lincs::ImproveProfilesWithAccuracyHeuristicOnGpu::improve_profiles,
      (bp::arg("self"), "model_indexes_begin", "model_indexes_end"),
      "Overrides the base method."
    )
  ;
  #endif  // LINCS_HAS_NVCC

  bp::class_<lincs::ReinitializeLeastAccurate, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy>>(
    "ReinitializeLeastAccurate",
    "The breeding strategy described in Olivier Sobrie's PhD thesis: re-initializes ``count`` in-progress models.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy&, unsigned>(
      (bp::arg("self"), "learning_data", "profiles_initialization_strategy", "count"),
      "Constructor. Keeps references to the profiles initialization strategy and the learning data."
    )[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3>>()])
    .def(
      "breed",
      &lincs::ReinitializeLeastAccurate::breed,
      (bp::arg("self")), 
      "Overrides the base method."
    )
  ;

  bp::class_<lincs::TerminateAtAccuracy, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAtAccuracy",
    "Termination strategy. Terminates the learning when the best model reaches a given accuracy.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, unsigned>(
      (bp::arg("self"), "learning_data", "target_accuracy"),
      "Constructor. Keeps a reference to the learning data."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "terminate",
      &lincs::TerminateAtAccuracy::terminate,
      (bp::arg("self")),
      "Overrides the base method."
    )
  ;

  bp::class_<lincs::TerminateAfterIterations, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterIterations",
    "Termination strategy. Terminates the learning after a given number of iterations.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, unsigned>(
      (bp::arg("self"), "learning_data", "max_iterations_count"),
      "Constructor. Keeps a reference to the learning data."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "terminate",
      &lincs::TerminateAfterIterations::terminate,
      (bp::arg("self")),
      "Overrides the base method."
    )
  ;

  bp::class_<lincs::TerminateAfterIterationsWithoutProgress, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterIterationsWithoutProgress",
    "Termination strategy. Terminates the learning after a given number of iterations without progress.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, unsigned>(
      (bp::arg("self"), "learning_data", "max_iterations_count"),
      "Constructor. Keeps a reference to the learning data."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "terminate",
      &lincs::TerminateAfterIterationsWithoutProgress::terminate,
      (bp::arg("self")),
      "Overrides the base method."
    )
  ;

  bp::class_<lincs::TerminateAfterSeconds, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterSeconds",
    "Termination strategy. Terminates the learning after a given duration.",
    bp::no_init
  )
    .def(bp::init<float>((bp::arg("self"), "max_seconds"), "Constructor."))
    .def(
      "terminate",
      &lincs::TerminateAfterSeconds::terminate,
      (bp::arg("self")),
      "Overrides the base method."
    )
  ;

  bp::class_<lincs::TerminateAfterSecondsWithoutProgress, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateAfterSecondsWithoutProgress",
    "Termination strategy. Terminates the learning after a given duration without progress.",
    bp::no_init
  )
    .def(bp::init<lincs::LearnMrsortByWeightsProfilesBreed::LearningData&, float>(
      (bp::arg("self"), "learning_data", "max_seconds"),
      "Constructor. Keeps a reference to the learning data."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "terminate",
      &lincs::TerminateAfterSecondsWithoutProgress::terminate,
      (bp::arg("self")),
      "Overrides the base method."
    )
  ;

  bp::class_<lincs::TerminateWhenAny, bp::bases<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy>>(
    "TerminateWhenAny",
    "Termination strategy. Terminates the learning when one or more termination strategies decide to terminate.",
    bp::no_init
  )
    .def(bp::init<std::vector<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy*>>(
      (bp::arg("self"), "termination_strategies"),
      "Constructor. Keeps references to each termination strategies."
    )[bp::with_custodian_and_ward<1, 2>()])
    .def(
      "terminate",
      &lincs::TerminateWhenAny::terminate,
      (bp::arg("self")),
      "Overrides the base method."
    )
  ;


  bp::class_<lincs::LearnUcncsBySatByCoalitionsUsingMinisat, boost::noncopyable>(
    "LearnUcncsBySatByCoalitionsUsingMinisat",
    "The \"SAT by coalitions\" approach to learn Uc-NCS models.",
    bp::no_init
  )
    .def(bp::init<const lincs::Problem&, const lincs::Alternatives&>(
      (bp::arg("self"), "problem", "learning_set"),
      "Constructor."
    )[bp::with_custodian_and_ward<1, 2 /* No reference kept on 'learning_set' => no custodian_and_ward */>()])
    .def("perform", &lincs::LearnUcncsBySatByCoalitionsUsingMinisat::perform, (bp::arg("self")), "Actually perform the learning and return the learned model.")
  ;

  bp::class_<lincs::LearnUcncsBySatBySeparationUsingMinisat, boost::noncopyable>(
    "LearnUcncsBySatBySeparationUsingMinisat",
    "The \"SAT by separation\" approach to learn Uc-NCS models.",
    bp::no_init
  )
    .def(bp::init<const lincs::Problem&, const lincs::Alternatives&>(
      (bp::arg("self"), "problem", "learning_set"),
      "Constructor."
    )[bp::with_custodian_and_ward<1, 2 /* No reference kept on 'learning_set' => no custodian_and_ward */>()])
    .def("perform", &lincs::LearnUcncsBySatBySeparationUsingMinisat::perform, (bp::arg("self")), "Actually perform the learning and return the learned model.")
  ;

  bp::class_<lincs::LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat, boost::noncopyable>(
    "LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat",
    "The \"max-SAT by coalitions\" approach to learn Uc-NCS models.",
    bp::no_init
  )
    .def(bp::init<const lincs::Problem&, const lincs::Alternatives&, unsigned, unsigned, unsigned>(
      (bp::arg("self"), "problem", "learning_set", bp::arg("nb_minimize_threads") = 0, bp::arg("timeout_fast_minimize") = 60, bp::arg("coef_minimize_time") = 2),
      "Constructor."
    )[bp::with_custodian_and_ward<1, 2 /* No reference kept on 'learning_set' => no custodian_and_ward */>()])
    .def("perform", &lincs::LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat::perform, (bp::arg("self")), "Actually perform the learning and return the learned model.")
  ;

  bp::class_<lincs::LearnUcncsByMaxSatBySeparationUsingEvalmaxsat, boost::noncopyable>(
    "LearnUcncsByMaxSatBySeparationUsingEvalmaxsat",
    "The \"max-SAT by separation\" approach to learn Uc-NCS models.",
    bp::no_init
  )
    .def(bp::init<const lincs::Problem&, const lincs::Alternatives&, unsigned, unsigned, unsigned>(
      (bp::arg("self"), "problem", "learning_set", bp::arg("nb_minimize_threads") = 0, bp::arg("timeout_fast_minimize") = 60, bp::arg("coef_minimize_time") = 2),
      "Constructor."
    )[bp::with_custodian_and_ward<1, 2 /* No reference kept on 'learning_set' => no custodian_and_ward */>()])
    .def("perform", &lincs::LearnUcncsByMaxSatBySeparationUsingEvalmaxsat::perform, (bp::arg("self")), "Actually perform the learning and return the learned model.")
  ;
}

}  // namespace lincs
