// Copyright 2023-2024 Vincent Jacques

#include "../classification.hpp"
#include "../generation.hpp"
#include "../vendored/pybind11/pybind11.h"
#include "../vendored/pybind11/stl.h"


namespace py = pybind11;
using namespace pybind11::literals;

namespace lincs {

void define_generation_functions(py::module& m) {
  m.def(
    "generate_classification_problem",
    &lincs::generate_classification_problem,
    "criteria_count"_a, "categories_count"_a, "random_seed"_a, "normalized_min_max"_a=true, "allowed_preference_directions"_a=std::vector{lincs::Criterion::PreferenceDirection::increasing}, "allowed_value_types"_a=std::vector{lincs::Criterion::ValueType::real},
    "Generate a :py:class:`Problem` with ``criteria_count`` criteria and ``categories_count`` categories."
  );
  m.def(
    "generate_mrsort_classification_model",
    &lincs::generate_mrsort_classification_model,
    "problem"_a, "random_seed"_a, "fixed_weights_sum"_a=std::optional<float>(),
    "Generate an MR-Sort model for the provided :py:class:`Problem`."
  );

  py::register_exception<lincs::BalancedAlternativesGenerationException>(m, "BalancedAlternativesGenerationException");

  m.def(
    "generate_classified_alternatives",
    &lincs::generate_classified_alternatives,
    "problem"_a, "model"_a, "alternatives_count"_a, "random_seed"_a, "max_imbalance"_a=std::optional<float>(),
    "Generate a set of ``alternatives_count`` pseudo-random alternatives for the provided :py:class:`Problem`, classified according to the provided :py:class:`Model`."
  );
  m.def(
    "misclassify_alternatives",
    &lincs::misclassify_alternatives,
    "problem"_a, "alternatives"_a, "count"_a, "random_seed"_a,
    "Misclassify ``count`` alternatives from the provided :py:class:`Alternatives`."
  );

  py::class_<lincs::ClassificationResult>(m, "ClassificationResult", "Return type for ``classify_alternatives``.")
    .def_readonly("changed", &lincs::ClassificationResult::changed, "Number of alternatives that were not in the same category before and after classification.")
    .def_readonly("unchanged", &lincs::ClassificationResult::unchanged, "Number of alternatives that were in the same category before and after classification.")
  ;
  m.def(
    "classify_alternatives",
    &lincs::classify_alternatives,
    "problem"_a, "model"_a, "alternatives"_a,
    "Classify the provided :py:class:`Alternatives` according to the provided :py:class:`Model`."
  );
}

}  // namespace lincs
