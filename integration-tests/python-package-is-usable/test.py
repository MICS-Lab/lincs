import io
import sys

import lincs


# @todo Remove details from this example, turn them into unit tests (see examples below)


criterion = lincs.Criterion("Physic grade", lincs.ValueType.real, lincs.CategoryCorrelation.growing)
print(criterion.name, criterion.value_type, criterion.category_correlation)
# Example of detail that should be changed into a unit test: the ability to change the name of a criterion
criterion.name = "Physics grade"
criterion.value_type = lincs.ValueType.real
criterion.category_correlation = lincs.CategoryCorrelation.growing

domain = lincs.Domain(
    [
        criterion,
        lincs.Criterion("Literature grade", lincs.ValueType.real, lincs.CategoryCorrelation.growing),
    ],
    (
        lincs.Category("Bad"),
        lincs.Category("Good"),
    ),
)

criterion = domain.criteria[0]
print(criterion.name, criterion.value_type, criterion.category_correlation)
print(domain.categories[0].name)
domain.categories[0].name = "Terrible"
domain.categories[1].name = "Ok, I guess"

buf = io.StringIO()
domain.dump(buf)
print(buf.getvalue().rstrip())

model = lincs.Model(domain, [lincs.Boundary([10.,10.], lincs.SufficientCoalitions(lincs.SufficientCoalitionsKind.weights, [0.4, 0.7]))])
model.dump(sys.stdout)

alternatives = lincs.AlternativesSet(domain, [lincs.Alternative("Alice", [11., 12.], "Good"), lincs.Alternative("Bob", [9., 11.], "Bad")])
alternatives.dump(sys.stdout)
