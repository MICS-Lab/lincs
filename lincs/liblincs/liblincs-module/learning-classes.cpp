// Copyright 2023-2024 Vincent Jacques

#include "../learning.hpp"
#include "../vendored/pybind11/pybind11.h"
#include "../vendored/pybind11/stl.h"


namespace py = pybind11;
using namespace pybind11::literals;

namespace lincs {

void define_learning_classes(py::module& m) {
  py::register_exception<lincs::LearningFailureException>(m, "LearningFailureException");

  auto learn_wbp_class = py::class_<lincs::LearnMrsortByWeightsProfilesBreed>(
    m,
    "LearnMrsortByWeightsProfilesBreed",
    "The approach described in Olivier Sobrie's PhD thesis to learn MR-Sort models."
  )
    .def("perform", &lincs::LearnMrsortByWeightsProfilesBreed::perform, "Actually perform the learning and return the learned model.")
  ;

  py::class_<lincs::PreprocessedLearningSet>(
    m,
    "PreprocessedLearningSet",
    "A representation of a learning set with its data normalized as ranks (unsigned integers)."
  )
    .def(
      py::init<const lincs::Problem&, const lincs::Alternatives&>(),
      "problem"_a, "learning_set"_a,
      "Constructor, pre-processing the learning set into a simpler form for learning.",
      py::keep_alive<1, 2>()
      // No reference kept on 'learning_set' => no py::keep_alive<1, 3>()
    )
    .def_readonly("criteria_count", &lincs::PreprocessedLearningSet::criteria_count, "Number of criteria in the :py:class:`Problem`.")
    .def_readonly("categories_count", &lincs::PreprocessedLearningSet::categories_count, "Number of categories in the :py:class:`Problem`.")
    .def_readonly("boundaries_count", &lincs::PreprocessedLearningSet::boundaries_count, "Number of boundaries in the :py:class:`Problem`, *i.e* ``categories_count - 1``.")
    .def_readonly("alternatives_count", &lincs::PreprocessedLearningSet::alternatives_count, "Number of alternatives in the ``learning_set``.")
    .def_readonly("single_peaked", &lincs::PreprocessedLearningSet::single_peaked, "Indexed by ``[criterion_index]``. Whether each criterion is single-peaked or not.")
    .def_readonly("values_counts", &lincs::PreprocessedLearningSet::values_counts, "Indexed by ``[criterion_index]``. Number of different values for each criterion, in the ``learning_set`` and min and max values for numerical criteria.")
    .def_readonly("performance_ranks", &lincs::PreprocessedLearningSet::performance_ranks, "Indexed by ``[criterion_index][alternative_index]``. Rank of each alternative in the ``learning_set`` for each criterion.")
    .def_readonly("assignments", &lincs::PreprocessedLearningSet::assignments, "Indexed by ``[alternative_index]``. Category index of each alternative in the ``learning_set``.")
  ;

  py::class_<lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned>(
    learn_wbp_class,
    "ModelsBeingLearned",
    "Data shared by all the strategies used in this learning."
  )
    .def(
      py::init<const lincs::PreprocessedLearningSet&, unsigned, unsigned>(),
      "preprocessed_learning_set"_a, "models_count"_a, "random_seed"_a,
      "Constructor, allocating but not initializing data about models about to be learned.",
      py::keep_alive<1, 2>()
    )
    .def_readonly("models_count", &lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned::models_count, "The number of in-progress models for this learning.")
    .def_readonly("random_generators", &lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned::random_generators, "Indexed by ``[model_index]``. Random number generators associated to each in-progress model.")
    .def_readonly("iteration_index", &lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned::iteration_index, "The index of the current iteration of the WPB algorithm.")
    .def_readonly("model_indexes", &lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned::model_indexes, "Indexed by ``0`` to ``models_count - 1``. Indexes of in-progress models ordered by increasing accuracy.")
    .def_readonly("accuracies", &lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned::accuracies, "Indexed by ``[model_index]``. Accuracy of each in-progress model.")
    .def_readonly("low_profile_ranks", &lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned::low_profile_ranks, "Indexed by ``[model_index][boundary_index][criterion_index]``. The current rank of each low profile, for each model and criterion.")
    .def_readonly("high_profile_rank_indexes", &lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned::high_profile_rank_indexes, "Indexed by ``[criterion_index]``. The index in ``high_profile_ranks``, for each single-peaked criterion.")
    .def_readonly("high_profile_ranks", &lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned::high_profile_ranks, "Indexed by ``[model_index][boundary_index][high_profile_rank_indexes[criterion_index]]``. The current rank of each high profile, for each model and single-peaked criterion.")
    .def_readonly("weights", &lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned::weights, "Indexed by ``[model_index][criterion_index]``. The current MR-Sort weight of each criterion for each model.")
    .def("get_best_accuracy", &lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned::get_best_accuracy, "Return the accuracy of the best model so far.")
    .def("get_best_model", &lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned::get_best_model, "Return the best model so far.")
  ;

  struct PyProfilesInitializationStrategy : lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy {
    using lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy::ProfilesInitializationStrategy;
    void initialize_profiles(const unsigned begin, const unsigned end) override {
      PYBIND11_OVERRIDE_PURE(
          void,
          lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy,
          initialize_profiles,
          begin, end
      );
    }
  };

  py::class_<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy, PyProfilesInitializationStrategy>(
    learn_wbp_class,
    "ProfilesInitializationStrategy",
    "Abstract base class for profiles initialization strategies."
  )
    .def(py::init<bool>(), py::arg("supports_single_peaked_criteria") = false)
    .def(
      "initialize_profiles",
      &lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy::initialize_profiles,
      "model_indexes_begin"_a, "model_indexes_end"_a,
      "Method to override. Should initialize all ``low_profile_ranks`` and ``high_profile_ranks`` of models at indexes in ``[model_indexes[i] for i in range(model_indexes_begin, model_indexes_end)]``."
    )
  ;

  struct PyWeightsOptimizationStrategy : lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy {
    using lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy::WeightsOptimizationStrategy;
    void optimize_weights(const unsigned begin, const unsigned end) override {
      PYBIND11_OVERRIDE_PURE(
          void,
          lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy,
          optimize_weights,
          begin, end
      );
    }
  };

  py::class_<lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy, PyWeightsOptimizationStrategy>(
    learn_wbp_class,
    "WeightsOptimizationStrategy",
    "Abstract base class for weights optimization strategies."
  )
    .def(py::init<bool>(), py::arg("supports_single_peaked_criteria") = false)
    .def(
      "optimize_weights",
      &lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy::optimize_weights,
      "model_indexes_begin"_a, "model_indexes_end"_a,
      "Method to override. Should optimize ``weights`` of models at indexes in ``[model_indexes[i] for i in range(model_indexes_begin, model_indexes_end)]``."
    )
  ;

  struct PyProfilesImprovementStrategy : lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy {
    using lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy::ProfilesImprovementStrategy;
    void improve_profiles(const unsigned begin, const unsigned end) override {
      PYBIND11_OVERRIDE_PURE(
          void,
          lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy,
          improve_profiles,
          begin, end
      );
    }
  };

  py::class_<lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy, PyProfilesImprovementStrategy>(
    learn_wbp_class,
    "ProfilesImprovementStrategy",
    "Abstract base class for profiles improvement strategies."
  )
    .def(py::init<bool>(), py::arg("supports_single_peaked_criteria") = false)
    .def(
      "improve_profiles",
      &lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy::improve_profiles,
      "model_indexes_begin"_a, "model_indexes_end"_a,
      "Method to override. Should improve ``low_profile_ranks`` and ``high_profile_ranks`` of models at indexes in ``[model_indexes[i] for i in range(model_indexes_begin, model_indexes_end)]``."
    )
  ;

  struct PyBreedingStrategy : lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy {
    using lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy::BreedingStrategy;
    void breed() override {
      PYBIND11_OVERRIDE_PURE(
          void,
          lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy,
          breed
      );
    }
  };

  py::class_<lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy, PyBreedingStrategy>(
    learn_wbp_class,
    "BreedingStrategy",
    "Abstract base class for breeding strategies."
  )
    .def(py::init<bool>(), py::arg("supports_single_peaked_criteria") = false)
    .def(
      "breed",
      &lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy::breed,
      "Method to override."
    )
  ;

  struct PyTerminationStrategy : lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
    using lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy::TerminationStrategy;
    bool terminate() override {
      PYBIND11_OVERRIDE_PURE(
          bool,
          lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy,
          terminate
      );
    }
  };

  py::class_<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy, PyTerminationStrategy>(
    learn_wbp_class,
    "TerminationStrategy",
    "Abstract base class for termination strategies."
  )
    .def(py::init<>())
    .def(
      "terminate",
      &lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy::terminate,
      "Method to override. Should return ``True`` if the learning should stop, ``False`` otherwise."
    )
  ;

  struct PyObserver : lincs::LearnMrsortByWeightsProfilesBreed::Observer {
    using lincs::LearnMrsortByWeightsProfilesBreed::Observer::Observer;
    void after_iteration() override {
      PYBIND11_OVERRIDE_PURE(
          void,
          lincs::LearnMrsortByWeightsProfilesBreed::Observer,
          after_iteration
      );
    }
    void before_return() override {
      PYBIND11_OVERRIDE_PURE(
          void,
          lincs::LearnMrsortByWeightsProfilesBreed::Observer,
          before_return
      );
    }
  };

  py::class_<lincs::LearnMrsortByWeightsProfilesBreed::Observer, PyObserver>(
    learn_wbp_class,
    "Observer",
    "Abstract base class for observation strategies."
  )
    .def(py::init<>())
    .def(
      "after_iteration",
      &lincs::LearnMrsortByWeightsProfilesBreed::Observer::after_iteration,
      "Method to override. Called after each iteration. Should not change anything in the learning data."
    )
    .def(
      "before_return",
      &lincs::LearnMrsortByWeightsProfilesBreed::Observer::before_return,
      "Method to override. Called just before returning the learned model. Should not change anything in the learning data."
    )
  ;

  learn_wbp_class
    .def(
      py::init<
        const PreprocessedLearningSet&,
        lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned&,
        lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy&,
        lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy&,
        lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy&,
        lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy&,
        lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy&,
        std::vector<lincs::LearnMrsortByWeightsProfilesBreed::Observer*>
      >(),
      "preprocessed_learning_set"_a,
      "models_being_learned"_a,
      "profiles_initialization_strategy"_a,
      "weights_optimization_strategy"_a,
      "profiles_improvement_strategy"_a,
      "breeding_strategy"_a,
      "termination_strategy"_a,
      "observers"_a=std::vector<lincs::LearnMrsortByWeightsProfilesBreed::Observer*>{},
      "Constructor accepting the strategies to use for each step of the learning.",
      py::keep_alive<1, 2>(),
      py::keep_alive<1, 3>(),
      py::keep_alive<1, 4>(),
      py::keep_alive<1, 5>(),
      py::keep_alive<1, 6>(),
      py::keep_alive<1, 7>(),
      py::keep_alive<1, 8>(),
      py::keep_alive<1, 9>()
    )
  ;

  py::class_<
    lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion,
    lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy
  >(
    m,
    "InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion",
    "The profiles initialization strategy described in Olivier Sobrie's PhD thesis."
  )
    .def(
      py::init<const lincs::PreprocessedLearningSet&, lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned&>(),
      "preprocessed_learning_set"_a, "models_being_learned"_a,
      "Constructor. Keeps a reference to the learning data.",
      py::keep_alive<1, 2>(),
      py::keep_alive<1, 3>()
    )
    .def(
      "initialize_profiles",
      &lincs::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles,
      "model_indexes_begin"_a, "model_indexes_end"_a,
      "Overrides the base method."
    )
  ;

  py::class_<
    lincs::OptimizeWeightsUsingGlop,
    lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy
  >(
    m,
    "OptimizeWeightsUsingGlop",
    "The weights optimization strategy described in Olivier Sobrie's PhD thesis. The linear program is solved using GLOP."
  )
    .def(
      py::init<const lincs::PreprocessedLearningSet&, lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned&>(),
      "preprocessed_learning_set"_a, "models_being_learned"_a,
      "Constructor. Keeps a reference to the learning data.",
      py::keep_alive<1, 2>(),
      py::keep_alive<1, 3>()
    )
    .def(
      "optimize_weights",
      &lincs::OptimizeWeightsUsingGlop::optimize_weights,
      "model_indexes_begin"_a, "model_indexes_end"_a,
      "Overrides the base method."
    )
  ;

  py::class_<
    lincs::OptimizeWeightsUsingAlglib,
    lincs::LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy
  >(
    m,
    "OptimizeWeightsUsingAlglib",
    "The weights optimization strategy described in Olivier Sobrie's PhD thesis. The linear program is solved using AlgLib."
  )
    .def(
      py::init<const lincs::PreprocessedLearningSet&, lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned&>(),
      "preprocessed_learning_set"_a, "models_being_learned"_a,
      "Constructor. Keeps a reference to the learning data.",
      py::keep_alive<1, 2>(),
      py::keep_alive<1, 3>()
    )
    .def(
      "optimize_weights",
      &lincs::OptimizeWeightsUsingAlglib::optimize_weights,
      "model_indexes_begin"_a, "model_indexes_end"_a,
      "Overrides the base method."
    )
  ;

  py::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnCpu,
    lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy
  >(
    m,
    "ImproveProfilesWithAccuracyHeuristicOnCpu",
    "The profiles improvement strategy described in Olivier Sobrie's PhD thesis. Run on the CPU."
  )
    .def(
      py::init<const lincs::PreprocessedLearningSet&, lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned&>(),
      "preprocessed_learning_set"_a, "models_being_learned"_a,
      "Constructor. Keeps a reference to the learning data.",
      py::keep_alive<1, 2>(),
      py::keep_alive<1, 3>()
    )
    .def(
      "improve_profiles",
      &lincs::ImproveProfilesWithAccuracyHeuristicOnCpu::improve_profiles,
      "model_indexes_begin"_a, "model_indexes_end"_a,
      "Overrides the base method."
    )
  ;

  #ifdef LINCS_HAS_NVCC
  py::class_<
    lincs::ImproveProfilesWithAccuracyHeuristicOnGpu,
    lincs::LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy
  >(
    m,
    "ImproveProfilesWithAccuracyHeuristicOnGpu",
    "The profiles improvement strategy described in Olivier Sobrie's PhD thesis. Run on the CUDA-capable GPU."
  )
    .def(
      py::init<const lincs::PreprocessedLearningSet&, lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned&>(),
      "preprocessed_learning_set"_a, "models_being_learned"_a,
      "Constructor. Keeps a reference to the learning data.",
      py::keep_alive<1, 2>(),
      py::keep_alive<1, 3>()
    )
    .def(
      "improve_profiles",
      &lincs::ImproveProfilesWithAccuracyHeuristicOnGpu::improve_profiles,
      "model_indexes_begin"_a, "model_indexes_end"_a,
      "Overrides the base method."
    )
  ;
  #endif  // LINCS_HAS_NVCC

  py::class_<
    lincs::ReinitializeLeastAccurate,
    lincs::LearnMrsortByWeightsProfilesBreed::BreedingStrategy
  >(
    m,
    "ReinitializeLeastAccurate",
    "The breeding strategy described in Olivier Sobrie's PhD thesis: re-initializes ``count`` in-progress models."
  )
    .def(
      py::init<lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned&, lincs::LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy&, unsigned>(),
      "models_being_learned"_a, "profiles_initialization_strategy"_a, "count"_a,
      "Constructor. Keeps references to the profiles initialization strategy and the learning data.",
      py::keep_alive<1, 2>(),
      py::keep_alive<1, 3>()
    )
    .def(
      "breed",
      &lincs::ReinitializeLeastAccurate::breed,
      "Overrides the base method."
    )
  ;

  py::class_<
    lincs::TerminateAtAccuracy,
    lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy
  >(
    m,
    "TerminateAtAccuracy",
    "Termination strategy. Terminates the learning when the best model reaches a given accuracy."
  )
    .def(
      py::init<lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned&, unsigned>(),
      "models_being_learned"_a, "target_accuracy"_a,
      "Constructor. Keeps a reference to the learning data.",
      py::keep_alive<1, 2>()
    )
    .def(
      "terminate",
      &lincs::TerminateAtAccuracy::terminate,
      "Overrides the base method."
    )
  ;

  py::class_<
    lincs::TerminateAfterIterations,
    lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy
  >(
    m,
    "TerminateAfterIterations",
    "Termination strategy. Terminates the learning after a given number of iterations."
  )
    .def(
      py::init<lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned&, unsigned>(),
      "models_being_learned"_a, "max_iterations_count"_a,
      "Constructor. Keeps a reference to the learning data.",
      py::keep_alive<1, 2>()
    )
    .def(
      "terminate",
      &lincs::TerminateAfterIterations::terminate,
      "Overrides the base method."
    )
  ;

  py::class_<
    lincs::TerminateAfterIterationsWithoutProgress,
    lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy
  >(
    m,
    "TerminateAfterIterationsWithoutProgress",
    "Termination strategy. Terminates the learning after a given number of iterations without progress."
  )
    .def(
      py::init<lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned&, unsigned>(),
      "models_being_learned"_a, "max_iterations_count"_a,
      "Constructor. Keeps a reference to the learning data.",
      py::keep_alive<1, 2>()
    )
    .def(
      "terminate",
      &lincs::TerminateAfterIterationsWithoutProgress::terminate,
      "Overrides the base method."
    )
  ;

  py::class_<
    lincs::TerminateAfterSeconds,
    lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy
  >(
    m,
    "TerminateAfterSeconds",
    "Termination strategy. Terminates the learning after a given duration."
  )
    .def(
      py::init<float>(),
      "max_seconds"_a,
      "Constructor."
    )
    .def(
      "terminate",
      &lincs::TerminateAfterSeconds::terminate,
      "Overrides the base method."
    )
  ;

  py::class_<
    lincs::TerminateAfterSecondsWithoutProgress,
    lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy
  >(
    m,
    "TerminateAfterSecondsWithoutProgress",
    "Termination strategy. Terminates the learning after a given duration without progress."
  )
    .def(
      py::init<lincs::LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned&, float>(),
      "models_being_learned"_a, "max_seconds"_a,
      "Constructor. Keeps a reference to the learning data.",
      py::keep_alive<1, 2>()
    )
    .def(
      "terminate",
      &lincs::TerminateAfterSecondsWithoutProgress::terminate,
      "Overrides the base method."
    )
  ;

  py::class_<
    lincs::TerminateWhenAny,
    lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy
  >(
    m,
    "TerminateWhenAny",
    "Termination strategy. Terminates the learning when one or more termination strategies decide to terminate."
  )
    .def(
      py::init<std::vector<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy*>>(),
      "termination_strategies"_a,
      "Constructor. Keeps references to each termination strategies.",
      py::keep_alive<1, 2>()
    )
    .def(
      "terminate",
      &lincs::TerminateWhenAny::terminate,
      "Overrides the base method."
    )
  ;


  py::class_<lincs::LearnUcncsBySatByCoalitionsUsingMinisat>(
    m,
    "LearnUcncsBySatByCoalitionsUsingMinisat",
    "The \"SAT by coalitions\" approach to learn Uc-NCS models."
  )
    .def(
      py::init<const lincs::Problem&, const lincs::Alternatives&>(),
      "problem"_a, "learning_set"_a,
      "Constructor.",
      py::keep_alive<1, 2>()
      // No py::keep_alive<1, 3>()
    )
    .def("perform", &lincs::LearnUcncsBySatByCoalitionsUsingMinisat::perform, "Actually perform the learning and return the learned model.")
  ;

  py::class_<lincs::LearnUcncsBySatBySeparationUsingMinisat>(
    m,
    "LearnUcncsBySatBySeparationUsingMinisat",
    "The \"SAT by separation\" approach to learn Uc-NCS models."
  )
    .def(
      py::init<const lincs::Problem&, const lincs::Alternatives&>(),
      "problem"_a, "learning_set"_a,
      "Constructor.",
      py::keep_alive<1, 2>()
      // No py::keep_alive<1, 3>()
    )
    .def("perform", &lincs::LearnUcncsBySatBySeparationUsingMinisat::perform, "Actually perform the learning and return the learned model.")
  ;

  py::class_<lincs::LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    m,
    "LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat",
    "The \"max-SAT by coalitions\" approach to learn Uc-NCS models."
  )
    .def(
      py::init<const lincs::Problem&, const lincs::Alternatives&, unsigned, unsigned, unsigned>(),
      "problem"_a, "learning_set"_a, "nb_minimize_threads"_a=0, "timeout_fast_minimize"_a=60, "coef_minimize_time"_a=2,
      "Constructor.",
      py::keep_alive<1, 2>()
      // No py::keep_alive<1, 3>()
    )
    .def("perform", &lincs::LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat::perform, "Actually perform the learning and return the learned model.")
  ;

  py::class_<lincs::LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    m,
    "LearnUcncsByMaxSatBySeparationUsingEvalmaxsat",
    "The \"max-SAT by separation\" approach to learn Uc-NCS models."
  )
    .def(
      py::init<const lincs::Problem&, const lincs::Alternatives&, unsigned, unsigned, unsigned>(),
      "problem"_a, "learning_set"_a, "nb_minimize_threads"_a=0, "timeout_fast_minimize"_a=60, "coef_minimize_time"_a=2,
      "Constructor.",
      py::keep_alive<1, 2>()
      // No py::keep_alive<1, 3>()
    )
    .def("perform", &lincs::LearnUcncsByMaxSatBySeparationUsingEvalmaxsat::perform, "Actually perform the learning and return the learned model.")
  ;
}

}  // namespace lincs
