# Copyright 2023-2024 Vincent Jacques

"""
The ``lincs`` package
=====================

This is the main module for the *lincs* library.
It contains general information (version, GPU availability, *etc.*) and items of general usage (*e.g.* the exception for invalid data).
"""

# General exceptions
from liblincs import DataValidationException, LearningFailureException

# General utilities
from liblincs import UniformRandomBitsGenerator

# Classification
from . import classification

# General information
__version__ = "1.1.0a6"
has_gpu = hasattr(classification, "ImproveProfilesWithAccuracyHeuristicOnGpu")

try:
    del visualization
except NameError:
    pass

try:
    del description
except NameError:
    pass

try:
    del command_line_interface
except NameError:
    pass
