# Copyright 2023 Vincent Jacques

import sys

import lincs


problem = lincs.Problem(
    [
        lincs.Criterion.make_real("Physics grade", lincs.Criterion.PreferenceDirection.increasing, 0, 1),
        lincs.Criterion.make_real("Literature grade", lincs.Criterion.PreferenceDirection.increasing, 0, 1),
    ],
    (
        lincs.Category("Bad"),
        lincs.Category("Good"),
    ),
)
problem.dump(sys.stdout)

print()

model = lincs.Model(
    problem,
    [lincs.AcceptedValues.make_real_thresholds([10.]), lincs.AcceptedValues.make_real_thresholds([10.])],
    [lincs.SufficientCoalitions.make_weights([0.4, 0.7])],
)
model.dump(problem, sys.stdout)

print()

alternatives = lincs.Alternatives(problem, [lincs.Alternative("Alice", [11., 12.], 1), lincs.Alternative("Bob", [9., 11.], 0)])
alternatives.dump(problem, sys.stdout)
