# Copyright 2023 Vincent Jacques

import io
import sys

import lincs


problem = lincs.Problem(
    [
        lincs.Criterion("Physics grade", lincs.ValueType.real, lincs.CategoryCorrelation.growing),
        lincs.Criterion("Literature grade", lincs.ValueType.real, lincs.CategoryCorrelation.growing),
    ],
    (
        lincs.Category("Bad"),
        lincs.Category("Good"),
    ),
)
problem.dump(sys.stdout)

model = lincs.Model(problem, [lincs.Boundary([10.,10.], lincs.SufficientCoalitions(lincs.SufficientCoalitions.weights, [0.4, 0.7]))])
model.dump(sys.stdout)

alternatives = lincs.Alternatives(problem, [lincs.Alternative("Alice", [11., 12.], 1), lincs.Alternative("Bob", [9., 11.], 0)])
alternatives.dump(problem, sys.stdout)
