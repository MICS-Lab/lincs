# Copyright 2023 Vincent Jacques

import sys

import lincs


problem = lincs.Problem(
    [
        lincs.Criterion("Physics grade", lincs.Criterion.ValueType.real, lincs.Criterion.CategoryCorrelation.growing, 0, 1),
        lincs.Criterion("Literature grade", lincs.Criterion.ValueType.real, lincs.Criterion.CategoryCorrelation.growing, 0, 1),
    ],
    (
        lincs.Category("Bad"),
        lincs.Category("Good"),
    ),
)
problem.dump(sys.stdout)

print()

model = lincs.Model(problem, [lincs.Model.Boundary([10.,10.], lincs.SufficientCoalitions(lincs.SufficientCoalitions.weights, [0.4, 0.7]))])
model.dump(problem, sys.stdout)

print()

alternatives = lincs.Alternatives(problem, [lincs.Alternative("Alice", [11., 12.], 1), lincs.Alternative("Bob", [9., 11.], 0)])
alternatives.dump(problem, sys.stdout)
