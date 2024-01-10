# Copyright 2023 Vincent Jacques
# @todo(Project management, v1.1) Update copyright notices everywhere

"""
The ``lincs`` package
=====================

This is the main module for the *lincs* library.
It contains general information (version, GPU availability, *etc.*) and items of general usage (*e.g.* the exception for invalid data).
"""

# General exceptions
from liblincs import DataValidationException, LearningFailureException

# Classification
from . import classification

# General information
__version__ = "1.1.0a0"
has_gpu = hasattr(classification, "ImproveProfilesWithAccuracyHeuristicOnGpu")
