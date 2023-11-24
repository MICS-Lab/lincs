# Copyright 2023 Vincent Jacques

import sys

import lincs


problem = lincs.Problem(
    [
        lincs.Criterion("Physics grade", lincs.Criterion.RealValues(lincs.Criterion.PreferenceDirection.increasing, 0, 1)),
        lincs.Criterion("Literature grade", lincs.Criterion.RealValues(lincs.Criterion.PreferenceDirection.increasing, 0, 1)),
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
    [lincs.AcceptedValues(lincs.AcceptedValues.RealThresholds([10.])), lincs.AcceptedValues(lincs.AcceptedValues.RealThresholds([10.]))],
    [lincs.SufficientCoalitions(lincs.SufficientCoalitions.Weights([0.4, 0.7]))],
)
model.dump(problem, sys.stdout)

print()

alternatives = lincs.Alternatives(
    problem,
    [
        lincs.Alternative("Alice", [lincs.Performance.make_real(11.), lincs.Performance.make_real(12.)], 1),
        lincs.Alternative("Bob", [lincs.Performance.make_real(9.), lincs.Performance.make_real(11.)], 0),
    ],
)
alternatives.dump(problem, sys.stdout)
