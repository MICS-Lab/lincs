# Copyright 2023 Vincent Jacques

import io
import sys

import lincs


domain = lincs.Domain(
    [
        lincs.Criterion("Physics grade", lincs.ValueType.real, lincs.CategoryCorrelation.growing),
        lincs.Criterion("Literature grade", lincs.ValueType.real, lincs.CategoryCorrelation.growing),
    ],
    (
        lincs.Category("Bad"),
        lincs.Category("Good"),
    ),
)
domain.dump(sys.stdout)

model = lincs.Model(domain, [lincs.Boundary([10.,10.], lincs.SufficientCoalitions(lincs.SufficientCoalitionsKind.weights, [0.4, 0.7]))])
model.dump(sys.stdout)

alternatives = lincs.Alternatives(domain, [lincs.Alternative("Alice", [11., 12.], "Good"), lincs.Alternative("Bob", [9., 11.], "Bad")])
alternatives.dump(sys.stdout)
