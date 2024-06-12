.. WARNING: this file is generated from 'doc-sources/reference/lincs.yml'. MANUAL EDITS WILL BE LOST.

.. module:: lincs

    The ``lincs`` package
    =====================

    This is the main module for the *lincs* library.
    It contains general information (version, GPU availability, *etc.*) and items of general usage (*e.g.* the exception for invalid data).

    .. data:: __version__
        :type: str

        The version of *lincs*, as a string in `Version Specifier <https://packaging.python.org/en/latest/specifications/version-specifiers/>`_ format.

    .. data:: has_gpu
        :type: bool

        ``True`` if *lincs* was built with CUDA support.

    .. exception:: DataValidationException

        Raised by constructors when called with invalid data. ``ex.args[0]`` gives a human-readable description of the error.

    .. exception:: LearningFailureException

        Raised by learning algorithms when they can't reach their objective.

    .. class:: UniformRandomBitsGenerator

        Random number generator.

        .. method:: __call__() -> int

            Generate the next pseudo-random integer.

    .. module:: lincs.classification

        The ``lincs.classification`` module
        -----------------------------------

        This module contains everything related to classification.

        .. class:: Criterion

            A classification criterion, to be used in a classification :py:class:`Problem`.

            .. method:: __init__(name: str, values: RealValues)

                Constructor for real-valued criterion.

            .. method:: __init__(name: str, values: IntegerValues)
                :noindex:

                Constructor for integer-valued criterion.

            .. method:: __init__(name: str, values: EnumeratedValues)
                :noindex:

                Constructor for criterion with enumerated values.

            .. property:: name
                :type: str

                The name of the criterion.

            .. class:: ValueType

                The different types of values for a criterion.

                .. property:: real
                    :classmethod:
                    :type: lincs.classification.Criterion.ValueType

                    Real values.

                .. property:: integer
                    :classmethod:
                    :type: lincs.classification.Criterion.ValueType

                    Integer values.

                .. property:: enumerated
                    :classmethod:
                    :type: lincs.classification.Criterion.ValueType

                    Enumerated values.

            .. property:: value_type
                :type: ValueType

                The type of values for this criterion.

            .. property:: is_real
                :type: bool

                ``True`` if the criterion is real-valued.

            .. property:: is_integer
                :type: bool

                ``True`` if the criterion is integer-valued.

            .. property:: is_enumerated
                :type: bool

                ``True`` if the criterion takes enumerated values.

            .. class:: PreferenceDirection

                What values are preferred for a criterion.

                .. property:: increasing
                    :classmethod:
                    :type: lincs.classification.Criterion.PreferenceDirection

                    For criteria where higher numerical values are known to be better.

                .. property:: decreasing
                    :classmethod:
                    :type: lincs.classification.Criterion.PreferenceDirection

                    For criteria where lower numerical values are known to be better.

                .. property:: single_peaked
                    :classmethod:
                    :type: lincs.classification.Criterion.PreferenceDirection

                    For criteria where intermediate numerical values are known to be better.

                .. property:: isotone
                    :classmethod:
                    :type: lincs.classification.Criterion.PreferenceDirection

                    Synonym for ``increasing``.

                .. property:: antitone
                    :classmethod:
                    :type: lincs.classification.Criterion.PreferenceDirection

                    Synonym for ``decreasing``.

            .. class:: RealValues

                Descriptor of the real values allowed for a criterion.

                .. method:: __init__(preference_direction: PreferenceDirection, min_value: float, max_value: float)

                    Parameters map exactly to attributes with identical names.

                .. property:: min_value
                    :type: float

                    The minimum value allowed for this criterion.

                .. property:: max_value
                    :type: float

                    The maximum value allowed for this criterion.

                .. property:: preference_direction
                    :type: PreferenceDirection

                    The preference direction for this criterion.

                .. property:: is_increasing
                    :type: bool

                    ``True`` if the criterion has increasing preference direction.

                .. property:: is_decreasing
                    :type: bool

                    ``True`` if the criterion has decreasing preference direction.

                .. property:: is_single_peaked
                    :type: bool

                    ``True`` if the criterion has single-peaked preference direction.

            .. property:: real_values
                :type: RealValues

                Descriptor of the real values allowed for this criterion, accessible if ``is_real``.

            .. class:: IntegerValues

                Descriptor of the integer values allowed for a criterion.

                .. method:: __init__(preference_direction: PreferenceDirection, min_value: int, max_value: int)

                    Parameters map exactly to attributes with identical names.

                .. property:: min_value
                    :type: float

                    The minimum value allowed for this criterion.

                .. property:: max_value
                    :type: float

                    The maximum value allowed for this criterion.

                .. property:: preference_direction
                    :type: PreferenceDirection

                    The preference direction for this criterion.

                .. property:: is_increasing
                    :type: bool

                    ``True`` if the criterion has increasing preference direction.

                .. property:: is_decreasing
                    :type: bool

                    ``True`` if the criterion has decreasing preference direction.

                .. property:: is_single_peaked
                    :type: bool

                    ``True`` if the criterion has single-peaked preference direction.

            .. property:: integer_values
                :type: IntegerValues

                Descriptor of the integer values allowed for this criterion, accessible if ``is_integer``.

            .. class:: EnumeratedValues

                Descriptor of the enumerated values allowed for a criterion.

                .. method:: __init__(ordered_values: list[str])

                    Parameters map exactly to attributes with identical names.

                .. method:: get_value_rank(value: str) -> int

                    Get the rank of a given value.

                .. property:: ordered_values
                    :type: list[str]

                    The values for this criterion, from the worst to the best.

            .. property:: enumerated_values
                :type: EnumeratedValues

                Descriptor of the enumerated values allowed for this criterion, accessible if ``is_enumerated``.

        .. class:: Category

            A category of a classification :py:class:`Problem`.

            .. method:: __init__(name: str)

                Parameters map exactly to attributes with identical names.

            .. property:: name
                :type: str

                The name of this category.

        .. class:: Problem

            A classification problem, with criteria and categories.

            .. method:: __init__(criteria: list[Criterion], ordered_categories: list[Category])

                Parameters map exactly to attributes with identical names.

            .. property:: criteria
                :type: list[Criterion]

                The criteria of this problem.

            .. property:: ordered_categories
                :type: list[Category]

                The categories of this problem, from the worst to the best.

            .. method:: dump(out: object)

                Dump the problem to the provided ``.write``-supporting file-like object, in YAML format.

            .. method:: load(in: object) -> Problem
                :staticmethod:

                Load a problem from the provided ``.read``-supporting file-like object, in YAML format.

            .. data:: JSON_SCHEMA
                :type: str

                The JSON schema defining the format used by ``dump`` and ``load``, as a string.

        .. class:: AcceptedValues

            The values accepted by a model for a criterion.

            .. method:: __init__(values: RealThresholds)

                Constructor for thresholds on a real-valued criterion.

            .. method:: __init__(values: IntegerThresholds)
                :noindex:

                Constructor for thresholds on an integer-valued criterion.

            .. method:: __init__(values: EnumeratedThresholds)
                :noindex:

                Constructor for thresholds on an enumerated criterion.

            .. method:: __init__(values: RealIntervals)
                :noindex:

                Constructor for intervals on a real-valued criterion.

            .. method:: __init__(values: IntegerIntervals)
                :noindex:

                Constructor for intervals on an integer-valued criterion.

            .. property:: value_type
                :type: ValueType

                The type of values for the corresponding criterion.

            .. property:: is_real
                :type: bool

                ``True`` if the corresponding criterion is real-valued.

            .. property:: is_integer
                :type: bool

                ``True`` if the corresponding criterion is integer-valued.

            .. property:: is_enumerated
                :type: bool

                ``True`` if the corresponding criterion takes enumerated values.

            .. class:: Kind

                The different kinds of descriptors for accepted values.

                .. property:: thresholds
                    :classmethod:
                    :type: lincs.classification.AcceptedValues.Kind

                    A threshold for each category.

                .. property:: intervals
                    :classmethod:
                    :type: lincs.classification.AcceptedValues.Kind

                    An interval for each category.

            .. property:: kind
                :type: AcceptedValues.Kind

                The kind of descriptor for these accepted values.

            .. property:: is_thresholds
                :type: bool

                ``True`` if the descriptor is a set of thresholds.

            .. property:: is_intervals
                :type: bool

                ``True`` if the descriptor is a set of intervals.

            .. class:: RealThresholds

                Descriptor for thresholds for an real-valued criterion.

                .. method:: __init__(thresholds: list[Optional[float]])

                    Parameters map exactly to attributes with identical names.

                .. property:: thresholds
                    :type: list[Optional[float]]

                    The thresholds for this descriptor.

            .. property:: real_thresholds
                :type: RealThresholds

                Descriptor of the real thresholds, accessible if ``is_real and is_thresholds``.

            .. class:: IntegerThresholds

                Descriptor for thresholds for an integer-valued criterion.

                .. method:: __init__(thresholds: list[Optional[int]])

                    Parameters map exactly to attributes with identical names.

                .. property:: thresholds
                    :type: list[Optional[int]]

                    The thresholds for this descriptor.

            .. property:: integer_thresholds
                :type: IntegerThresholds

                Descriptor of the integer thresholds, accessible if ``is_integer and is_thresholds``.

            .. class:: EnumeratedThresholds

                Descriptor for thresholds for a criterion taking enumerated values.

                .. method:: __init__(thresholds: list[Optional[str]])

                    Parameters map exactly to attributes with identical names.

                .. property:: thresholds
                    :type: list[Optional[str]]

                    The thresholds for this descriptor.

            .. property:: enumerated_thresholds
                :type: EnumeratedThresholds

                Descriptor of the enumerated thresholds, accessible if ``is_enumerated and is_thresholds``.

            .. class:: RealIntervals

                Descriptor for intervals for an real-valued criterion.

                .. method:: __init__(intervals: list[Optional[tuple[float, float]]])

                    Parameters map exactly to attributes with identical names.

                .. property:: intervals
                    :type: list[Optional[tuple[float, float]]]

                    The intervals for this descriptor.

            .. property:: real_intervals
                :type: RealIntervals

                Descriptor of the real intervals, accessible if ``is_real and is_intervals``.

            .. class:: IntegerIntervals

                Descriptor for intervals for an integer-valued criterion.

                .. method:: __init__(intervals: list[Optional[tuple[int, int]]])

                    Parameters map exactly to attributes with identical names.

                .. property:: intervals
                    :type: list[Optional[tuple[int, int]]]

                    The intervals for this descriptor.

            .. property:: integer_intervals
                :type: IntegerIntervals

                Descriptor of the integer intervals, accessible if ``is_integer and is_intervals``.

        .. class:: SufficientCoalitions

            The coalitions of sufficient criteria to accept an alternative in a category.

            .. method:: __init__(weights: Weights)

                Constructor for sufficient coalitions defined by weights.

            .. method:: __init__(roots: Roots)
                :noindex:

                Constructor for sufficient coalitions defined by roots.

            .. class:: Kind

                The different kinds of descriptors for sufficient coalitions.

                .. property:: weights
                    :classmethod:
                    :type: lincs.classification.SufficientCoalitions.Kind

                    For sufficient coalitions described by criterion weights.

                .. property:: roots
                    :classmethod:
                    :type: lincs.classification.SufficientCoalitions.Kind

                    For sufficient coalitions described by the roots of their upset.

            .. property:: kind
                :type: SufficientCoalitions.Kind

                The kind of descriptor for these sufficient coalitions.

            .. property:: is_weights
                :type: bool

                ``True`` if the descriptor is a set of weights.

            .. property:: is_roots
                :type: bool

                ``True`` if the descriptor is a set of roots.

            .. class:: Weights

                Descriptor for sufficient coalitions defined by weights.

                .. method:: __init__(criterion_weights: list[float])

                    Parameters map exactly to attributes with identical names.

                .. property:: criterion_weights
                    :type: list[float]

                    The weights for each criterion.

            .. property:: weights
                :type: Weights

                Descriptor of the weights, accessible if ``is_weights``.

            .. class:: Roots

                Descriptor for sufficient coalitions defined by roots.

                .. method:: __init__(problem: Problem, upset_roots: list[list[int]])

                    Parameters map exactly to attributes with identical names.

                .. property:: upset_roots
                    :type: list[list[int]]

                    The roots of the upset of sufficient coalitions.

            .. property:: roots
                :type: Roots

                Descriptor of the roots, accessible if ``is_roots``.

        .. class:: Model

            An NCS classification model.

            .. method:: __init__(problem: Problem, accepted_values: list[AcceptedValues], sufficient_coalitions: list[SufficientCoalitions])

                The :py:class:`Model` being initialized must correspond to the given :py:class:`Problem`. Other parameters map exactly to attributes with identical names.

            .. property:: accepted_values
                :type: list[AcceptedValues]

                The accepted values for each criterion.

            .. property:: sufficient_coalitions
                :type: list[SufficientCoalitions]

                The sufficient coalitions for each category.

            .. method:: dump(problem: Problem, out: object)

                Dump the model to the provided ``.write``-supporting file-like object, in YAML format.

            .. method:: load(problem: Problem, in: object) -> Model
                :staticmethod:

                Load a model for the provided ``Problem``, from the provided ``.read``-supporting file-like object, in YAML format.

            .. data:: JSON_SCHEMA
                :type: str

                The JSON schema defining the format used by ``dump`` and ``load``, as a string.

        .. class:: Performance

            The performance of an alternative on a criterion.

            .. method:: __init__(performance: Real)

                Constructor for a real-valued performance.

            .. method:: __init__(performance: Integer)
                :noindex:

                Constructor for an integer-valued performance.

            .. method:: __init__(performance: Enumerated)
                :noindex:

                Constructor for an enumerated performance.

            .. property:: value_type
                :type: ValueType

                The type of values for the corresponding criterion.

            .. property:: is_real
                :type: bool

                ``True`` if the corresponding criterion is real-valued.

            .. property:: is_integer
                :type: bool

                ``True`` if the corresponding criterion is integer-valued.

            .. property:: is_enumerated
                :type: bool

                ``True`` if the corresponding criterion takes enumerated values.

            .. class:: Real

                A performance for a real-valued criterion.

                .. method:: __init__(value: float)

                    Parameters map exactly to attributes with identical names.

                .. property:: value
                    :type: float

                    The numerical value of the real performance.

            .. property:: real
                :type: Real

                The real performance, accessible if ``is_real``.

            .. class:: Integer

                A performance for an integer-valued criterion.

                .. method:: __init__(value: int)

                    Parameters map exactly to attributes with identical names.

                .. property:: value
                    :type: int

                    The numerical value of the integer performance.

            .. property:: integer
                :type: Integer

                The integer performance, accessible if ``is_integer``.

            .. class:: Enumerated

                A performance for a criterion taking enumerated values.

                .. method:: __init__(value: str)

                    Parameters map exactly to attributes with identical names.

                .. property:: value
                    :type: str

                    The string value of the enumerated performance.

            .. property:: enumerated
                :type: Enumerated

                The enumerated performance, accessible if ``is_enumerated``.

        .. class:: Alternative

            An alternative, with its performance on each criterion, maybe classified.

            .. method:: __init__(name: str, profile: list[Performance], category_index: Optional[int]=None)

                Parameters map exactly to attributes with identical names.

            .. property:: name
                :type: str

                The name of the alternative.

            .. property:: profile
                :type: list[Performance]

                The performance profile of the alternative.

            .. property:: category_index
                :type: Optional[int]

                The index of the category of the alternative, if it is classified.

        .. class:: Alternatives

            A set of alternatives, maybe classified.

            .. method:: __init__(problem: Problem, alternatives: list[Alternative])

                The :py:class:`Alternatives` being initialized must correspond to the given :py:class:`Problem`. Other parameters map exactly to attributes with identical names.

            .. property:: alternatives
                :type: list[Alternative]

                The :py:class:`Alternative` objects in this set.

            .. method:: dump(problem: Problem, out: object)

                Dump the set of alternatives to the provided ``.write``-supporting file-like object, in CSV format.

            .. method:: load(problem: Problem, in: object) -> Alternatives
                :staticmethod:

                Load a set of alternatives (classified or not) from the provided ``.read``-supporting file-like object, in CSV format.

        .. function:: generate_problem(criteria_count: int, categories_count: int, random_seed: int, normalized_min_max: bool=True, allowed_preference_directions: list[PreferenceDirection]=[PreferenceDirection.increasing], allowed_value_types: list[ValueType]=[ValueType.real]) -> Problem

            Generate a :py:class:`Problem` with ``criteria_count`` criteria and ``categories_count`` categories.

        .. function:: generate_mrsort_model(problem: Problem, random_seed: int, fixed_weights_sum: Optional[float]=None) -> Model

            Generate an MR-Sort model for the provided :py:class:`Problem`.

        .. exception:: BalancedAlternativesGenerationException

            Raised by ``generate_alternatives`` when it fails to find alternatives to balance the categories.

        .. function:: generate_alternatives(problem: Problem, model: Model, alternatives_count: int, random_seed: int, max_imbalance: Optional[float]=None) -> Alternatives

            Generate a set of ``alternatives_count`` pseudo-random alternatives for the provided :py:class:`Problem`, classified according to the provided :py:class:`Model`.

        .. function:: misclassify_alternatives(problem: Problem, alternatives: Alternatives, count: int, random_seed: int)

            Misclassify ``count`` alternatives from the provided :py:class:`Alternatives`.

        .. class:: LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat

            The "max-SAT by coalitions" approach to learn Uc-NCS models.

            .. method:: __init__(problem: Problem, learning_set: Alternatives, nb_minimize_threads: int=0, timeout_fast_minimize: int=60, coef_minimize_time: int=2)

                Constructor.

            .. method:: perform() -> Model

                Actually perform the learning and return the learned model.

        .. class:: LearnUcncsByMaxSatBySeparationUsingEvalmaxsat

            The "max-SAT by separation" approach to learn Uc-NCS models.

            .. method:: __init__(problem: Problem, learning_set: Alternatives, nb_minimize_threads: int=0, timeout_fast_minimize: int=60, coef_minimize_time: int=2)

                Constructor.

            .. method:: perform() -> Model

                Actually perform the learning and return the learned model.

        .. class:: LearnUcncsBySatByCoalitionsUsingMinisat

            The "SAT by coalitions" approach to learn Uc-NCS models.

            .. method:: __init__(problem: Problem, learning_set: Alternatives)

                Constructor.

            .. method:: perform() -> Model

                Actually perform the learning and return the learned model.

        .. class:: LearnUcncsBySatBySeparationUsingMinisat

            The "SAT by separation" approach to learn Uc-NCS models.

            .. method:: __init__(problem: Problem, learning_set: Alternatives)

                Constructor.

            .. method:: perform() -> Model

                Actually perform the learning and return the learned model.

        .. class:: LearnMrsortByWeightsProfilesBreed

            The approach described in Olivier Sobrie's PhD thesis to learn MR-Sort models.

            .. method:: __init__(learning_data: LearningData, profiles_initialization_strategy: ProfilesInitializationStrategy, weights_optimization_strategy: WeightsOptimizationStrategy, profiles_improvement_strategy: ProfilesImprovementStrategy, breeding_strategy: BreedingStrategy, termination_strategy: TerminationStrategy, observers: list[Observer]=[])

                Constructor accepting the strategies to use for each step of the learning.

            .. class:: LearningData

                Data shared by all the strategies used in this learning.

                .. method:: __init__(problem: Problem, learning_set: Alternatives, models_count: int, random_seed: int)

                    Constructor, pre-processing the learning set into a simpler form for strategies.

                .. property:: criteria_count
                    :type: int

                    Number of criteria in the :py:class:`Problem`.

                .. property:: categories_count
                    :type: int

                    Number of categories in the :py:class:`Problem`.

                .. property:: boundaries_count
                    :type: int

                    Number of boundaries in the :py:class:`Problem`, *i.e* ``categories_count - 1``.

                .. property:: alternatives_count
                    :type: int

                    Number of alternatives in the ``learning_set``.

                .. property:: values_counts
                    :type: list[int]

                    Indexed by ``[criterion_index]``. Number of different values for each criterion, in the ``learning_set`` and min and max values for numerical criteria.

                .. property:: performance_ranks
                    :type: list[list[int]]

                    Indexed by ``[criterion_index][alternative_index]``. Rank of each alternative in the ``learning_set`` for each criterion.

                .. property:: assignments
                    :type: list[int]

                    Indexed by ``[alternative_index]``. Category index of each alternative in the ``learning_set``.

                .. property:: models_count
                    :type: int

                    The number of in-progress models for this learning.

                .. property:: random_generators
                    :type: list[UniformRandomBitsGenerator]

                    Indexed by ``[model_index]``. Random number generators associated to each in-progress model.

                .. property:: iteration_index
                    :type: int

                    The index of the current iteration of the WPB algorithm.

                .. property:: model_indexes
                    :type: list[int]

                    Indexed by ``0`` to ``models_count - 1``. Indexes of in-progress models ordered by increasing accuracy.

                .. property:: weights
                    :type: list[list[int]]

                    Indexed by ``[model_index][criterion_index]``. The current MR-Sort weight of each criterion for each model.

                .. property:: low_profile_ranks
                    :type: list[list[list[int]]]

                    Indexed by ``[model_index][profile_index][criterion_index]``. The current rank of each low profile, for each model and criterion.

                .. property:: high_profile_ranks
                    :type: list[list[list[int]]]

                    Indexed by ``[model_index][profile_index][criterion_index]``. The current rank of each high profile, for each model and criterion.

                .. property:: accuracies
                    :type: list[int]

                    Indexed by ``[model_index]``. Accuracy of each in-progress model.

                .. method:: get_best_accuracy() -> int

                    Return the accuracy of the best model so far.

                .. method:: get_best_model() -> Model

                    Return the best model so far.

            .. class:: ProfilesInitializationStrategy

                Abstract base class for profiles initialization strategies.

                .. method:: initialize_profiles(model_indexes_begin: int, model_indexes_end: int)

                    Method to override. Should initialize all ``low_profile_ranks`` and ``high_profile_ranks`` of models at indexes in ``[model_indexes[i] for i in range(model_indexes_begin, model_indexes_end)]``.

            .. class:: WeightsOptimizationStrategy

                Abstract base class for weights optimization strategies.

                .. method:: optimize_weights(model_indexes_begin: int, model_indexes_end: int)

                    Method to override. Should optimize ``weights`` of models at indexes in ``[model_indexes[i] for i in range(model_indexes_begin, model_indexes_end)]``.

            .. class:: ProfilesImprovementStrategy

                Abstract base class for profiles improvement strategies.

                .. method:: improve_profiles(model_indexes_begin: int, model_indexes_end: int)

                    Method to override. Should improve ``low_profile_ranks`` and ``high_profile_ranks`` of models at indexes in ``[model_indexes[i] for i in range(model_indexes_begin, model_indexes_end)]``.

            .. class:: BreedingStrategy

                Abstract base class for breeding strategies.

                .. method:: breed()

                    Method to override.

            .. class:: TerminationStrategy

                Abstract base class for termination strategies.

                .. method:: terminate() -> bool

                    Method to override. Should return ``True`` if the learning should stop, ``False`` otherwise.

            .. class:: Observer

                Abstract base class for observation strategies.

                .. method:: after_iteration()

                    Method to override. Called after each iteration. Should not change anything in the learning data.

                .. method:: before_return()

                    Method to override. Called just before returning the learned model. Should not change anything in the learning data.

            .. method:: perform() -> Model

                Actually perform the learning and return the learned model.

        .. class:: InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion

            The profiles initialization strategy described in Olivier Sobrie's PhD thesis.

            .. method:: __init__(learning_data: LearningData)

                Constructor. Keeps a reference to the learning data.

            .. method:: initialize_profiles(model_indexes_begin: int, model_indexes_end: int)

                Overrides the base method.

        .. class:: OptimizeWeightsUsingAlglib

            The weights optimization strategy described in Olivier Sobrie's PhD thesis. The linear program is solved using AlgLib.

            .. method:: __init__(learning_data: LearningData)

                Constructor. Keeps a reference to the learning data.

            .. method:: optimize_weights(model_indexes_begin: int, model_indexes_end: int)

                Overrides the base method.

        .. class:: OptimizeWeightsUsingGlop

            The weights optimization strategy described in Olivier Sobrie's PhD thesis. The linear program is solved using GLOP.

            .. method:: __init__(learning_data: LearningData)

                Constructor. Keeps a reference to the learning data.

            .. method:: optimize_weights(model_indexes_begin: int, model_indexes_end: int)

                Overrides the base method.

        .. class:: ImproveProfilesWithAccuracyHeuristicOnCpu

            The profiles improvement strategy described in Olivier Sobrie's PhD thesis. Run on the CPU.

            .. method:: __init__(learning_data: LearningData)

                Constructor. Keeps a reference to the learning data.

            .. method:: improve_profiles(model_indexes_begin: int, model_indexes_end: int)

                Overrides the base method.

        .. class:: ImproveProfilesWithAccuracyHeuristicOnGpu

            The profiles improvement strategy described in Olivier Sobrie's PhD thesis. Run on the CUDA-capable GPU.

            .. method:: __init__(learning_data: LearningData)

                Constructor. Keeps a reference to the learning data.

            .. method:: improve_profiles(model_indexes_begin: int, model_indexes_end: int)

                Overrides the base method.

        .. class:: ReinitializeLeastAccurate

            The breeding strategy described in Olivier Sobrie's PhD thesis: re-initializes ``count`` in-progress models.

            .. method:: __init__(learning_data: LearningData, profiles_initialization_strategy: ProfilesInitializationStrategy, count: int)

                Constructor. Keeps references to the profiles initialization strategy and the learning data.

            .. method:: breed()

                Overrides the base method.

        .. class:: TerminateAfterIterations

            Termination strategy. Terminates the learning after a given number of iterations.

            .. method:: __init__(learning_data: LearningData, max_iterations_count: int)

                Constructor. Keeps a reference to the learning data.

            .. method:: terminate() -> bool

                Overrides the base method.

        .. class:: TerminateAfterIterationsWithoutProgress

            Termination strategy. Terminates the learning after a given number of iterations without progress.

            .. method:: __init__(learning_data: LearningData, max_iterations_count: int)

                Constructor. Keeps a reference to the learning data.

            .. method:: terminate() -> bool

                Overrides the base method.

        .. class:: TerminateAfterSeconds

            Termination strategy. Terminates the learning after a given duration.

            .. method:: __init__(max_seconds: float)

                Constructor.

            .. method:: terminate() -> bool

                Overrides the base method.

        .. class:: TerminateAfterSecondsWithoutProgress

            Termination strategy. Terminates the learning after a given duration without progress.

            .. method:: __init__(learning_data: LearningData, max_seconds: float)

                Constructor. Keeps a reference to the learning data.

            .. method:: terminate() -> bool

                Overrides the base method.

        .. class:: TerminateAtAccuracy

            Termination strategy. Terminates the learning when the best model reaches a given accuracy.

            .. method:: __init__(learning_data: LearningData, target_accuracy: int)

                Constructor. Keeps a reference to the learning data.

            .. method:: terminate() -> bool

                Overrides the base method.

        .. class:: TerminateWhenAny

            Termination strategy. Terminates the learning when one or more termination strategies decide to terminate.

            .. method:: __init__(termination_strategies: list[TerminationStrategy])

                Constructor. Keeps references to each termination strategies.

            .. method:: terminate() -> bool

                Overrides the base method.

        .. class:: ClassificationResult

            Return type for ``classify_alternatives``.

            .. property:: changed
                :type: int

                Number of alternatives that were not in the same category before and after classification.

            .. property:: unchanged
                :type: int

                Number of alternatives that were in the same category before and after classification.

        .. function:: classify_alternatives(problem: Problem, model: Model, alternatives: Alternatives) -> ClassificationResult

            Classify the provided :py:class:`Alternatives` according to the provided :py:class:`Model`.

        .. function:: describe_model(problem: lincs.classification.Problem, model: lincs.classification.Model) -> Iterable[str]

            Generate a human-readable description of a classification model.

        .. function:: describe_problem(problem: lincs.classification.Problem) -> Iterable[str]

            Generate a human-readable description of a classification problem.

        .. function:: visualize_model(problem: lincs.classification.Problem, model: lincs.classification.Model, alternatives: Iterable[lincs.classification.Alternative], axes: matplotlib.axes._axes.Axes)

            Create a visual representation of a classification model and classified alternatives, using Matplotlib.

