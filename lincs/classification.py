# Copyright 2023-2024 Vincent Jacques

"""
The ``lincs.classification`` module
-----------------------------------

This module contains everything related to classification.
"""

# @todo(Feature, later) Provide __repr__ and __str__ where applicable

# I/O
from liblincs import Criterion, Category, Problem
from liblincs import AcceptedValues, SufficientCoalitions, Model
from liblincs import Performance, Alternative, Alternatives

# Generation (incl. misclassification)
from liblincs import BalancedAlternativesGenerationException
from liblincs import generate_classification_problem as generate_problem
from liblincs import generate_mrsort_classification_model as generate_mrsort_model
from liblincs import generate_classified_alternatives as generate_alternatives
from liblincs import misclassify_alternatives

# Classification
from liblincs import ClassificationResult
from liblincs import classify_alternatives

# Learning - weights-profiles-breed
from liblincs import LearnMrsortByWeightsProfilesBreed
from liblincs import InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion
from liblincs import OptimizeWeightsUsingGlop, OptimizeWeightsUsingAlglib
from liblincs import ImproveProfilesWithAccuracyHeuristicOnCpu
try:
    from liblincs import ImproveProfilesWithAccuracyHeuristicOnGpu
except ImportError:
    pass
from liblincs import ReinitializeLeastAccurate
from liblincs import TerminateAtAccuracy
from liblincs import TerminateAfterSeconds, TerminateAfterSecondsWithoutProgress
from liblincs import TerminateAfterIterations, TerminateAfterIterationsWithoutProgress
from liblincs import TerminateWhenAny

# Learning - SAT by coalitions
from liblincs import LearnUcncsBySatByCoalitionsUsingMinisat

# Learning - SAT by separation
from liblincs import LearnUcncsBySatBySeparationUsingMinisat

# Learning - max-SAT by coalitions
from liblincs import LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat

# Learning - max-SAT by separation
from liblincs import LearnUcncsByMaxSatBySeparationUsingEvalmaxsat

from .visualization import visualize_classification_model as visualize_model
from .description import describe_classification_problem as describe_problem
from .description import describe_classification_model as describe_model

# @todo(Feature, later) Accept learning and training set as Pandas DataFrame?
