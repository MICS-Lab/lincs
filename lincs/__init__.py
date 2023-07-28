# Copyright 2023 Vincent Jacques

# I/O
from liblincs import Criterion, Category, Problem

from liblincs import SufficientCoalitions, Model
from liblincs import Alternative, Alternatives

# Generation (incl. misclassification)
from liblincs import generate_classification_problem, generate_mrsort_classification_model, generate_classified_alternatives, misclassify_alternatives

# Classification
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
from liblincs import TerminateAtAccuracy, TerminateAfterSeconds, TerminateAfterIterations, TerminateWhenAny

# Learning - SAT by coalitions
from liblincs import LearnUcncsBySatByCoalitionsUsingMinisat

# Learning - SAT by separation
from liblincs import LearnUcncsBySatBySeparationUsingMinisat

# @todo Accept learning and training set as Pandas DataFrame?
