.. WARNING: this file is generated from 'doc-sources/reference/lincs.yml'. MANUAL EDITS WILL BE LOST.

.. module:: lincs

    The ``lincs`` package
    =====================

    This is the main module for the *lincs* library.
    It contains general information (version, GPU availability, *etc.*) and items of general usage (*e.g.* the exception for invalid data).

    .. data:: __version__
        :type: str

        The version of *lincs*, as a string in `SemVer <https://semver.org/>`_ format

    .. data:: has_gpu
        :type: bool

        ``True`` if *lincs* was compiled with CUDA support

    .. exception:: DataValidationException

        Raised by constructors when called with invalid data. ``ex.args[0]`` gives a human-readable description of the error

    .. exception:: LearningFailureException

        Raised by learning algorithms when they can't reach their objective

    .. module:: lincs.classification

        The ``lincs.classification`` module
        -----------------------------------

        This module contains everything related to classification.

        .. class:: Criterion

            A classification criterion, to be used in a classification ``Problem``

            .. method:: __init__(name: str, values: RealValues)

                Constructor for real-valued criterion

            .. method:: __init__(name: str, values: IntegerValues)
                :noindex:

                Constructor for integer-valued criterion

            .. method:: __init__(name: str, values: EnumeratedValues)
                :noindex:

                Constructor for criterion with enumerated values

            .. property:: name
                :type: str

                The name of the criterion

            .. class:: ValueType

                The different types of values for a criterion

                .. property:: real
                    :classmethod:
                    :type: lincs.classification.Criterion.ValueType

                    Real values

                .. property:: integer
                    :classmethod:
                    :type: lincs.classification.Criterion.ValueType

                    Integer values

                .. property:: enumerated
                    :classmethod:
                    :type: lincs.classification.Criterion.ValueType

                    Enumerated values

            .. property:: value_type
                :type: ValueType

                The type of values for this criterion

            .. property:: is_real
                :type: bool

                ``True`` if the criterion is real-valued

            .. property:: is_integer
                :type: bool

                ``True`` if the criterion is integer-valued

            .. property:: is_enumerated
                :type: bool

                ``True`` if the criterion takes enumerated values

            .. class:: PreferenceDirection

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Criterion.PreferenceDirection

                .. property:: increasing
                    :classmethod:
                    :type: lincs.classification.Criterion.PreferenceDirection

                    For criteria where higher numerical values are known to be better

                .. property:: decreasing
                    :classmethod:
                    :type: lincs.classification.Criterion.PreferenceDirection

                    For criteria where lower numerical values are known to be better

                .. property:: isotone
                    :classmethod:
                    :type: lincs.classification.Criterion.PreferenceDirection

                    Synonym for ``increasing``

                .. property:: antitone
                    :classmethod:
                    :type: lincs.classification.Criterion.PreferenceDirection

                    Synonym for ``decreasing``

            .. class:: RealValues

                Descriptor of the real values allowed for a criterion

                .. method:: __init__(preference_direction: PreferenceDirection, min_value: float, max_value: float)

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Criterion.RealValues.__init__

                .. property:: min_value
                    :type: float

                    The minimum value allowed for this criterion

                .. property:: max_value
                    :type: float

                    The maximum value allowed for this criterion

                .. property:: preference_direction
                    :type: PreferenceDirection

                    The preference direction for this criterion

                .. property:: is_increasing
                    :type: bool

                    ``True`` if the criterion has increasing preference direction

                .. property:: is_decreasing
                    :type: bool

                    ``True`` if the criterion has decreasing preference direction

            .. property:: real_values
                :type: RealValues

                Descriptor of the real values allowed for this criterion, accessible if ``is_real``

            .. class:: IntegerValues

                Descriptor of the integer values allowed for a criterion

                .. method:: __init__(preference_direction: PreferenceDirection, min_value: int, max_value: int)

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Criterion.IntegerValues.__init__

                .. property:: min_value
                    :type: float

                    The minimum value allowed for this criterion

                .. property:: max_value
                    :type: float

                    The maximum value allowed for this criterion

                .. property:: preference_direction
                    :type: PreferenceDirection

                    The preference direction for this criterion

                .. property:: is_increasing
                    :type: bool

                    ``True`` if the criterion has increasing preference direction

                .. property:: is_decreasing
                    :type: bool

                    ``True`` if the criterion has decreasing preference direction

            .. property:: integer_values
                :type: IntegerValues

                Descriptor of the integer values allowed for this criterion, accessible if ``is_integer``

            .. class:: EnumeratedValues

                Descriptor of the enumerated values allowed for a criterion

                .. method:: __init__(ordered_values: Iterable[str])

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Criterion.EnumeratedValues.__init__

                .. method:: get_value_rank(value: str) -> int

                    Get the rank of a given value

                .. property:: ordered_values
                    :type: Iterable[str]

                    The values for this criterion, from the worst to the best

            .. property:: enumerated_values
                :type: EnumeratedValues

                Descriptor of the enumerated values allowed for this criterion, accessible if ``is_enumerated``

        .. class:: Category

            A category of a classification ``Problem``

            .. method:: __init__(name: str)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Category.__init__

            .. property:: name
                :type: str

                The name of this category

        .. class:: Problem

            A classification problem, with criteria and categories

            .. method:: __init__(criteria: Iterable[Criterion], categories: Iterable[Category])

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Problem.__init__

            .. property:: criteria
                :type: Iterable[Criterion]

                The criteria of this problem

            .. property:: ordered_categories
                :type: Iterable[Category]

                The categories of this problem, from the worst to the best

            .. method:: dump(out: object)

                Dump the problem to the provided ``.write``-supporting file-like object, in YAML format

            .. method:: load(in: object) -> Problem
                :staticmethod:

                Load a problem from the provided ``.read``-supporting file-like object, in YAML format

            .. data:: JSON_SCHEMA
                :type: str

                The JSON schema defining the format used by ``dump`` and ``load``, as a string

        .. class:: AcceptedValues

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.AcceptedValues

            .. method:: __init__(values: RealThresholds)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.AcceptedValues.__init__

            .. method:: __init__(values: IntegerThresholds)
                :noindex:

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.AcceptedValues.__init__

            .. method:: __init__(values: EnumeratedThresholds)
                :noindex:

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.AcceptedValues.__init__

            .. property:: value_type
                :type: ValueType

                The type of values for the corresponding criterion

            .. property:: is_real
                :type: bool

                ``True`` if the corresponding criterion is real-valued

            .. property:: is_integer
                :type: bool

                ``True`` if the corresponding criterion is integer-valued

            .. property:: is_enumerated
                :type: bool

                ``True`` if the corresponding criterion takes enumerated values

            .. class:: Kind

                The different kinds of descriptors for accepted values

                .. property:: thresholds
                    :classmethod:
                    :type: lincs.classification.AcceptedValues.Kind

                    A threshold for each category

            .. property:: kind
                :type: AcceptedValues.Kind

                The kind of descriptor for these accepted values

            .. property:: is_thresholds
                :type: bool

                ``True`` if the descriptor is a set of thresholds

            .. class:: RealThresholds

                Descriptor for thresholds for an real-valued criterion

                .. method:: __init__(thresholds: Iterable[float])

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.AcceptedValues.RealThresholds.__init__

                .. property:: thresholds
                    :type: Iterable[float]

                    The thresholds for this descriptor

            .. property:: real_thresholds
                :type: RealThresholds

                Descriptor of the real thresholds, accessible if ``is_real and is_thresholds``

            .. class:: IntegerThresholds

                Descriptor for thresholds for an integer-valued criterion

                .. method:: __init__(thresholds: Iterable[int])

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.AcceptedValues.IntegerThresholds.__init__

                .. property:: thresholds
                    :type: Iterable[int]

                    The thresholds for this descriptor

            .. property:: integer_thresholds
                :type: IntegerThresholds

                Descriptor of the integer thresholds, accessible if ``is_integer and is_thresholds``

            .. class:: EnumeratedThresholds

                Descriptor for thresholds for a criterion taking enumerated values

                .. method:: __init__(thresholds: Iterable[str])

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.AcceptedValues.EnumeratedThresholds.__init__

                .. property:: thresholds
                    :type: Iterable[str]

                    The thresholds for this descriptor

            .. property:: enumerated_thresholds
                :type: EnumeratedThresholds

                Descriptor of the enumerated thresholds, accessible if ``is_enumerated and is_thresholds``

        .. class:: SufficientCoalitions

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.SufficientCoalitions

            .. method:: __init__(weights: Weights)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.SufficientCoalitions.__init__

            .. method:: __init__(roots: Roots)
                :noindex:

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.SufficientCoalitions.__init__

            .. class:: Kind

                The different kinds of descriptors for sufficient coalitions

                .. property:: weights
                    :classmethod:
                    :type: lincs.classification.SufficientCoalitions.Kind

                    For sufficient coalitions described by criterion weights

                .. property:: roots
                    :classmethod:
                    :type: lincs.classification.SufficientCoalitions.Kind

                    For sufficient coalitions described by the roots of their upset

            .. property:: kind
                :type: SufficientCoalitions.Kind

                The kind of descriptor for these sufficient coalitions

            .. property:: is_weights
                :type: bool

                ``True`` if the descriptor is a set of weights

            .. property:: is_roots
                :type: bool

                ``True`` if the descriptor is a set of roots

            .. class:: Weights

                Descriptor for sufficient coalitions defined by weights

                .. method:: __init__(criterion_weights: Iterable[float])

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.SufficientCoalitions.Weights.__init__

                .. property:: criterion_weights
                    :type: Iterable[float]

                    The weights for each criterion

            .. property:: weights
                :type: Weights

                Descriptor of the weights, accessible if ``is_weights``

            .. class:: Roots

                Descriptor for sufficient coalitions defined by roots

                .. method:: __init__(criteria_count: int, upset_roots: object)

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.SufficientCoalitions.Roots.__init__

                .. property:: upset_roots
                    :type: Iterable[Iterable[int]]

                    The roots of the upset of sufficient coalitions

            .. property:: roots
                :type: Roots

                Descriptor of the roots, accessible if ``is_roots``

        .. class:: Model

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Model

            .. method:: __init__(problem: Problem, accepted_values: Iterable[AcceptedValues], sufficient_coalitions: Iterable[SufficientCoalitions])

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Model.__init__

            .. property:: accepted_values
                :type: Iterable[AcceptedValues]

                The accepted values for each criterion

            .. property:: sufficient_coalitions
                :type: Iterable[SufficientCoalitions]

                The sufficient coalitions for each category

            .. method:: dump(problem: Problem, out: object)

                Dump the model to the provided ``.write``-supporting file-like object, in YAML format

            .. method:: load(problem: Problem, in: object) -> Model
                :staticmethod:

                Load a model for the provided ``Problem``, from the provided ``.read``-supporting file-like object, in YAML format

            .. data:: JSON_SCHEMA
                :type: str

                The JSON schema defining the format used by ``dump`` and ``load``, as a string

        .. class:: Performance

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance

            .. method:: __init__(performance: RealPerformance)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.__init__

            .. method:: __init__(performance: IntegerPerformance)
                :noindex:

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.__init__

            .. method:: __init__(performance: EnumeratedPerformance)
                :noindex:

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.__init__

            .. property:: value_type
                :type: ValueType

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.value_type

            .. property:: is_real
                :type: bool

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.is_real

            .. property:: is_integer
                :type: bool

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.is_integer

            .. property:: is_enumerated
                :type: bool

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.is_enumerated

            .. class:: RealPerformance

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.RealPerformance

                .. method:: __init__(value: float)

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.RealPerformance.__init__

                .. property:: value
                    :type: float

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.RealPerformance.value

            .. property:: real
                :type: RealPerformance

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.real

            .. class:: IntegerPerformance

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.IntegerPerformance

                .. method:: __init__(value: int)

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.IntegerPerformance.__init__

                .. property:: value
                    :type: int

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.IntegerPerformance.value

            .. property:: integer
                :type: IntegerPerformance

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.integer

            .. class:: EnumeratedPerformance

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.EnumeratedPerformance

                .. method:: __init__(value: str)

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.EnumeratedPerformance.__init__

                .. property:: value
                    :type: str

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.EnumeratedPerformance.value

            .. property:: enumerated
                :type: EnumeratedPerformance

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Performance.enumerated

        .. class:: Alternative

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Alternative

            .. method:: __init__(name: str, profile: Iterable[Performance] [, category: object=None])

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Alternative.__init__

            .. property:: name
                :type: str

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Alternative.name

            .. property:: profile
                :type: Iterable[Performance]

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Alternative.profile

            .. property:: category_index
                :type: Optional[int]

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Alternative.category_index

        .. class:: Alternatives

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Alternatives

            .. method:: __init__(problem: Problem, alternatives: Iterable[Alternative])

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Alternatives.__init__

            .. property:: alternatives
                :type: Iterable[Alternative]

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.Alternatives.alternatives

            .. method:: dump(problem: Problem, out: object)

                Dump the set of alternatives to the provided ``.write``-supporting file-like object, in CSV format.

            .. method:: load(problem: Problem, in: object) -> Alternatives
                :staticmethod:

                Load a set of alternatives (classified or not) from the provided ``.read``-supporting file-like object, in CSV format.

        .. function:: generate_problem(criteria_count: int, categories_count: int, random_seed: int [, normalized_min_max: bool=True [, allowed_preference_directions: Iterable[PreferenceDirection]=[] [, allowed_value_types: Iterable[ValueType]=[]]]]) -> Problem

            Generate a problem with ``criteria_count`` criteria and ``categories_count`` categories.

        .. function:: generate_mrsort_model(problem: Problem, random_seed: int [, fixed_weights_sum: object=None]) -> Model

            Generate an MR-Sort model for the provided ``Problem``.

        .. exception:: BalancedAlternativesGenerationException

            Raised by ``generate_classified_alternatives`` when it fails to find alternatives to balance the categories

        .. function:: generate_classified_alternatives(problem: Problem, model: Model, alternatives_count: int, random_seed: int [, max_imbalance: object=None]) -> Alternatives

            Generate a set of ``alternatives_count`` pseudo-random alternatives for the provided ``Problem``, classified according to the provided ``Model``.

        .. function:: misclassify_alternatives(problem: Problem, alternatives: Alternatives, count: int, random_seed: int)

            Misclassify ``count`` alternatives from the provided ``Alternatives``.

        .. class:: LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat

            .. method:: __init__(problem: Problem, learning_set: Alternatives)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat.__init__

            .. method:: perform() -> Model

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat.perform

        .. class:: LearnUcncsByMaxSatBySeparationUsingEvalmaxsat

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsByMaxSatBySeparationUsingEvalmaxsat

            .. method:: __init__(problem: Problem, learning_set: Alternatives)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsByMaxSatBySeparationUsingEvalmaxsat.__init__

            .. method:: perform() -> Model

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsByMaxSatBySeparationUsingEvalmaxsat.perform

        .. class:: LearnUcncsBySatByCoalitionsUsingMinisat

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsBySatByCoalitionsUsingMinisat

            .. method:: __init__(problem: Problem, learning_set: Alternatives)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsBySatByCoalitionsUsingMinisat.__init__

            .. method:: perform() -> Model

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsBySatByCoalitionsUsingMinisat.perform

        .. class:: LearnUcncsBySatBySeparationUsingMinisat

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsBySatBySeparationUsingMinisat

            .. method:: __init__(problem: Problem, learning_set: Alternatives)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsBySatBySeparationUsingMinisat.__init__

            .. method:: perform() -> Model

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnUcncsBySatBySeparationUsingMinisat.perform

        .. class:: LearnMrsortByWeightsProfilesBreed

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed

            .. method:: __init__(learning_data: LearningData, profiles_initialization_strategy: ProfilesInitializationStrategy, weights_optimization_strategy: WeightsOptimizationStrategy, profiles_improvement_strategy: ProfilesImprovementStrategy, breeding_strategy: BreedingStrategy, termination_strategy: TerminationStrategy)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.__init__

            .. method:: __init__(learning_data: LearningData, profiles_initialization_strategy: ProfilesInitializationStrategy, weights_optimization_strategy: WeightsOptimizationStrategy, profiles_improvement_strategy: ProfilesImprovementStrategy, breeding_strategy: BreedingStrategy, termination_strategy: TerminationStrategy, observers: object)
                :noindex:

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.__init__

            .. class:: LearningData

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.LearningData

                .. method:: __init__(problem: Problem, learning_set: Alternatives, models_count: int, random_seed: int)

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.LearningData.__init__

                .. method:: get_best_accuracy() -> int

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.LearningData.get_best_accuracy

                .. property:: iteration_index
                    :type: int

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.LearningData.iteration_index

            .. class:: ProfilesInitializationStrategy

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.ProfilesInitializationStrategy

                .. method:: initialize_profiles(model_indexes_begin: int, model_indexes_end: int)

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.ProfilesInitializationStrategy.initialize_profiles

            .. class:: WeightsOptimizationStrategy

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.WeightsOptimizationStrategy

                .. method:: optimize_weights()

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.WeightsOptimizationStrategy.optimize_weights

            .. class:: ProfilesImprovementStrategy

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.ProfilesImprovementStrategy

                .. method:: improve_profiles()

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.ProfilesImprovementStrategy.improve_profiles

            .. class:: BreedingStrategy

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.BreedingStrategy

                .. method:: breed()

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.BreedingStrategy.breed

            .. class:: TerminationStrategy

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.TerminationStrategy

                .. method:: terminate() -> bool

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.TerminationStrategy.terminate

            .. class:: Observer

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.Observer

                .. method:: after_iteration()

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.Observer.after_iteration

                .. method:: before_return()

                    .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.Observer.before_return

            .. method:: perform() -> Model

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.LearnMrsortByWeightsProfilesBreed.perform

        .. class:: InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion

            .. method:: __init__(learning_data: LearningData)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion.__init__

            .. method:: initialize_profiles(model_indexes_begin: int, model_indexes_end: int)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion.initialize_profiles

        .. class:: OptimizeWeightsUsingAlglib

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.OptimizeWeightsUsingAlglib

            .. method:: __init__(learning_data: LearningData)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.OptimizeWeightsUsingAlglib.__init__

            .. method:: optimize_weights()

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.OptimizeWeightsUsingAlglib.optimize_weights

        .. class:: OptimizeWeightsUsingGlop

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.OptimizeWeightsUsingGlop

            .. method:: __init__(learning_data: LearningData)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.OptimizeWeightsUsingGlop.__init__

            .. method:: optimize_weights()

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.OptimizeWeightsUsingGlop.optimize_weights

        .. class:: ImproveProfilesWithAccuracyHeuristicOnCpu

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ImproveProfilesWithAccuracyHeuristicOnCpu

            .. method:: __init__(learning_data: LearningData)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ImproveProfilesWithAccuracyHeuristicOnCpu.__init__

            .. method:: improve_profiles()

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ImproveProfilesWithAccuracyHeuristicOnCpu.improve_profiles

        .. class:: ImproveProfilesWithAccuracyHeuristicOnGpu

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ImproveProfilesWithAccuracyHeuristicOnGpu

            .. method:: __init__(learning_data: LearningData)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ImproveProfilesWithAccuracyHeuristicOnGpu.__init__

            .. method:: improve_profiles()

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ImproveProfilesWithAccuracyHeuristicOnGpu.improve_profiles

        .. class:: ReinitializeLeastAccurate

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ReinitializeLeastAccurate

            .. method:: __init__(learning_data: LearningData, profiles_initialization_strategy: ProfilesInitializationStrategy, count: int)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ReinitializeLeastAccurate.__init__

            .. method:: breed()

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ReinitializeLeastAccurate.breed

        .. class:: TerminateAfterIterations

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterIterations

            .. method:: __init__(learning_data: LearningData, max_iteration_index: int)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterIterations.__init__

            .. method:: terminate() -> bool

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterIterations.terminate

        .. class:: TerminateAfterIterationsWithoutProgress

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterIterationsWithoutProgress

            .. method:: __init__(learning_data: LearningData, max_iterations_count: int)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterIterationsWithoutProgress.__init__

            .. method:: terminate() -> bool

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterIterationsWithoutProgress.terminate

        .. class:: TerminateAfterSeconds

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterSeconds

            .. method:: __init__(max_seconds: float)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterSeconds.__init__

            .. method:: terminate() -> bool

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterSeconds.terminate

        .. class:: TerminateAfterSecondsWithoutProgress

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterSecondsWithoutProgress

            .. method:: __init__(learning_data: LearningData, max_seconds: float)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterSecondsWithoutProgress.__init__

            .. method:: terminate() -> bool

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAfterSecondsWithoutProgress.terminate

        .. class:: TerminateAtAccuracy

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAtAccuracy

            .. method:: __init__(learning_data: LearningData, target_accuracy: int)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAtAccuracy.__init__

            .. method:: terminate() -> bool

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateAtAccuracy.terminate

        .. class:: TerminateWhenAny

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateWhenAny

            .. method:: __init__(termination_strategies: object)

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateWhenAny.__init__

            .. method:: terminate() -> bool

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.TerminateWhenAny.terminate

        .. class:: ClassificationResult

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ClassificationResult

            .. property:: changed
                :type: int

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ClassificationResult.changed

            .. property:: unchanged
                :type: int

                .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.ClassificationResult.unchanged

        .. function:: classify_alternatives(problem: Problem, model: Model, alternatives: Alternatives) -> ClassificationResult

            Classify the provided ``Alternatives`` according to the provided ``Model``.

        .. function:: describe_model(problem: lincs.classification.Problem, model: lincs.classification.Model)

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.describe_model

        .. function:: describe_problem(problem: lincs.classification.Problem)

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.describe_problem

        .. function:: visualize_model(problem: lincs.classification.Problem, model: lincs.classification.Model, alternatives: lincs.classification.Alternatives, axes: matplotlib.axes._axes.Axes)

            .. @todo(Documentation, v1.1) Add a docstring to lincs.classification.visualize_model

