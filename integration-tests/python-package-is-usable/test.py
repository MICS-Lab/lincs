# Copyright 2023-2024 Vincent Jacques

import sys

import lincs


problem = lincs.classification.Problem(
    [
        lincs.classification.Criterion("Physics grade", lincs.classification.Criterion.RealValues(lincs.classification.Criterion.PreferenceDirection.increasing, 0, 20)),
        lincs.classification.Criterion("Literature grade", lincs.classification.Criterion.RealValues(lincs.classification.Criterion.PreferenceDirection.increasing, 0, 20)),
    ],
    (
        lincs.classification.Category("Bad"),
        lincs.classification.Category("Good"),
    ),
)
problem.dump(sys.stdout)

print()

model = lincs.classification.Model(
    problem,
    [lincs.classification.AcceptedValues(lincs.classification.AcceptedValues.RealThresholds([10.])), lincs.classification.AcceptedValues(lincs.classification.AcceptedValues.RealThresholds([10.]))],
    [lincs.classification.SufficientCoalitions(lincs.classification.SufficientCoalitions.Weights([0.4, 0.7]))],
)
model.dump(problem, sys.stdout)

print()

alternatives = lincs.classification.Alternatives(
    problem,
    [
        lincs.classification.Alternative(
            "Alice",
            [
                lincs.classification.Performance(lincs.classification.Performance.Real(11.)),
                lincs.classification.Performance(lincs.classification.Performance.Real(12.)),
            ],
            1,
        ),
        lincs.classification.Alternative(
            "Bob",
            [
                lincs.classification.Performance(lincs.classification.Performance.Real(9.)),
                lincs.classification.Performance(lincs.classification.Performance.Real(11.)),
            ],
            0,
        ),
    ],
)
alternatives.dump(problem, sys.stdout)
