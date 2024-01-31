.. WARNING: this file is generated from 'doc-sources/reference/lincs.yml'. MANUAL EDITS WILL BE LOST.

.. module:: lincs

    The ``lincs`` package
    =====================

    This is the main module for the *lincs* library.
    It contains general information (version, GPU availability, *etc.*) and items of general usage (*e.g.* the exception for invalid data).

    .. data:: __version__
        :type: str

        The version of *lincs*, as a string in `Version Specifier <https://packaging.python.org/en/latest/specifications/version-specifiers/>`_ format

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

            A classification criterion, to be used in a classification :py:class:`Problem`

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

                What values are preferred for a criterion

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

                    Parameters map exactly to attributes with identical names

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

                    Parameters map exactly to attributes with identical names

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

                    Parameters map exactly to attributes with identical names

                .. method:: get_value_rank(value: str) -> int

                    Get the rank of a given value

                .. property:: ordered_values
                    :type: Iterable[str]

                    The values for this criterion, from the worst to the best

            .. property:: enumerated_values
                :type: EnumeratedValues

                Descriptor of the enumerated values allowed for this criterion, accessible if ``is_enumerated``

        .. class:: Category

            A category of a classification :py:class:`Problem`

            .. method:: __init__(name: str)

                Parameters map exactly to attributes with identical names

            .. property:: name
                :type: str

                The name of this category

        .. class:: Problem

            A classification problem, with criteria and categories

            .. method:: __init__(criteria: Iterable[Criterion], ordered_categories: Iterable[Category])

                Parameters map exactly to attributes with identical names

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

            The values accepted by a model for a criterion

            .. method:: __init__(values: RealThresholds)

                Constructor for thresholds on a real-valued criterion

            .. method:: __init__(values: IntegerThresholds)
                :noindex:

                Constructor for thresholds on an integer-valued criterion

            .. method:: __init__(values: EnumeratedThresholds)
                :noindex:

                Constructor for thresholds on an enumerated criterion

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

                    Parameters map exactly to attributes with identical names

                .. property:: thresholds
                    :type: Iterable[float]

                    The thresholds for this descriptor

            .. property:: real_thresholds
                :type: RealThresholds

                Descriptor of the real thresholds, accessible if ``is_real and is_thresholds``

            .. class:: IntegerThresholds

                Descriptor for thresholds for an integer-valued criterion

                .. method:: __init__(thresholds: Iterable[int])

                    Parameters map exactly to attributes with identical names

                .. property:: thresholds
                    :type: Iterable[int]

                    The thresholds for this descriptor

            .. property:: integer_thresholds
                :type: IntegerThresholds

                Descriptor of the integer thresholds, accessible if ``is_integer and is_thresholds``

            .. class:: EnumeratedThresholds

                Descriptor for thresholds for a criterion taking enumerated values

                .. method:: __init__(thresholds: Iterable[str])

                    Parameters map exactly to attributes with identical names

                .. property:: thresholds
                    :type: Iterable[str]

                    The thresholds for this descriptor

            .. property:: enumerated_thresholds
                :type: EnumeratedThresholds

                Descriptor of the enumerated thresholds, accessible if ``is_enumerated and is_thresholds``

        .. class:: SufficientCoalitions

            The coalitions of sufficient criteria to accept an alternative in a category

            .. method:: __init__(weights: Weights)

                Constructor for sufficient coalitions defined by weights

            .. method:: __init__(roots: Roots)
                :noindex:

                Constructor for sufficient coalitions defined by roots

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

                    Parameters map exactly to attributes with identical names

                .. property:: criterion_weights
                    :type: Iterable[float]

                    The weights for each criterion

            .. property:: weights
                :type: Weights

                Descriptor of the weights, accessible if ``is_weights``

            .. class:: Roots

                Descriptor for sufficient coalitions defined by roots

                .. method:: __init__(problem: Problem, upset_roots: Iterable[Iterable[int]])

                    Parameters map exactly to attributes with identical names

                .. property:: upset_roots
                    :type: Iterable[Iterable[int]]

                    The roots of the upset of sufficient coalitions

            .. property:: roots
                :type: Roots

                Descriptor of the roots, accessible if ``is_roots``

        .. class:: Model

            An NCS classification model

            .. method:: __init__(problem: Problem, accepted_values: Iterable[AcceptedValues], sufficient_coalitions: Iterable[SufficientCoalitions])

                The :py:class:`Model` being initialized must correspond to the given :py:class:`Problem`. Other parameters map exactly to attributes with identical names

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

            The performance of an alternative on a criterion

            .. method:: __init__(performance: Real)

                Constructor for a real-valued performance

            .. method:: __init__(performance: Integer)
                :noindex:

                Constructor for an integer-valued performance

            .. method:: __init__(performance: Enumerated)
                :noindex:

                Constructor for an enumerated performance

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

            .. class:: Real

                A performance for a real-valued criterion

                .. method:: __init__(value: float)

                    Parameters map exactly to attributes with identical names

                .. property:: value
                    :type: float

                    The numerical value of the real performance

            .. property:: real
                :type: Real

                The real performance, accessible if ``is_real``

            .. class:: Integer

                A performance for an integer-valued criterion

                .. method:: __init__(value: int)

                    Parameters map exactly to attributes with identical names

                .. property:: value
                    :type: int

                    The numerical value of the integer performance

            .. property:: integer
                :type: Integer

                The integer performance, accessible if ``is_integer``

            .. class:: Enumerated

                A performance for a criterion taking enumerated values

                .. method:: __init__(value: str)

                    Parameters map exactly to attributes with identical names

                .. property:: value
                    :type: str

                    The string value of the enumerated performance

            .. property:: enumerated
                :type: Enumerated

                The enumerated performance, accessible if ``is_enumerated``

        .. class:: Alternative

            An alternative, with its performance on each criterion, maybe classified

            .. method:: __init__(name: str, profile: Iterable[Performance] [, category_index: object=None])

                Parameters map exactly to attributes with identical names

            .. property:: name
                :type: str

                The name of the alternative

            .. property:: profile
                :type: Iterable[Performance]

                The performance profile of the alternative

            .. property:: category_index
                :type: Optional[int]

                The index of the category of the alternative, if it is classified

        .. class:: Alternatives

            A set of alternatives, maybe classified

            .. method:: __init__(problem: Problem, alternatives: Iterable[Alternative])

                The :py:class:`Alternatives` being initialized must correspond to the given :py:class:`Problem`. Other parameters map exactly to attributes with identical names

            .. property:: alternatives
                :type: Iterable[Alternative]

                The :py:class:`Alternative` objects in this set

            .. method:: dump(problem: Problem, out: object)

                Dump the set of alternatives to the provided ``.write``-supporting file-like object, in CSV format.

            .. method:: load(problem: Problem, in: object) -> Alternatives
                :staticmethod:

                Load a set of alternatives (classified or not) from the provided ``.read``-supporting file-like object, in CSV format.

        .. function:: generate_problem(criteria_count: int, categories_count: int, random_seed: int [, normalized_min_max: bool=True [, allowed_preference_directions: Iterable[PreferenceDirection]=[] [, allowed_value_types: Iterable[ValueType]=[]]]]) -> Problem

            Generate a problem with ``criteria_count`` criteria and ``categories_count`` categories.

        .. function:: generate_mrsort_model(problem: Problem, random_seed: int [, fixed_weights_sum: Optional[float]=None]) -> Model

            Generate an MR-Sort model for the provided ``Problem``.

        .. exception:: BalancedAlternativesGenerationException

            Raised by ``generate_alternatives`` when it fails to find alternatives to balance the categories

        .. function:: generate_alternatives(problem: Problem, model: Model, alternatives_count: int, random_seed: int [, max_imbalance: Optional[float]=None]) -> Alternatives

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

            .. method:: __init__(learning_data: LearningData, profiles_initialization_strategy: ProfilesInitializationStrategy, weights_optimization_strategy: WeightsOptimizationStrategy, profiles_improvement_strategy: ProfilesImprovementStrategy, breeding_strategy: BreedingStrategy, termination_strategy: TerminationStrategy [, observers: Iterable[Observer]=[]])

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

            .. method:: __init__(termination_strategies: Iterable[TerminationStrategy])

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

