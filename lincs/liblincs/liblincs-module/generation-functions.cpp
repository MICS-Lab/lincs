// Copyright 2023 Vincent Jacques

#include <Python.h>
// https://bugs.python.org/issue36020#msg371558
#undef snprintf
#undef vsnprintf

#include <boost/python.hpp>

#include "../classification.hpp"
#include "../generation.hpp"


namespace bp = boost::python;

namespace lincs {

void define_generation_functions() {
  bp::def(
    "generate_classification_problem",
    &lincs::generate_classification_problem,
    (
      bp::arg("criteria_count"),
      "categories_count",
      "random_seed",
      bp::arg("normalized_min_max")=true,
      bp::arg("allowed_preference_directions")=std::vector{lincs::Criterion::PreferenceDirection::increasing},
      bp::arg("allowed_value_types")=std::vector{lincs::Criterion::ValueType::real}
    ),
    "Generate a problem with `criteria_count` criteria and `categories_count` categories."
  );
  bp::def(
    "generate_mrsort_classification_model",
    &lincs::generate_mrsort_classification_model,
    (bp::arg("problem"), "random_seed", bp::arg("fixed_weights_sum")=std::optional<float>()),
    "Generate an MR-Sort model for the provided `problem`."
  );

  PyObject* BalancedAlternativesGenerationException_wrapper = PyErr_NewException("liblincs.BalancedAlternativesGenerationException", PyExc_RuntimeError, NULL);
  bp::register_exception_translator<lincs::BalancedAlternativesGenerationException>(
    [BalancedAlternativesGenerationException_wrapper](const lincs::BalancedAlternativesGenerationException& e) {
      PyErr_SetString(BalancedAlternativesGenerationException_wrapper, e.what());
    }
  );
  bp::scope().attr("BalancedAlternativesGenerationException") = bp::handle<>(bp::borrowed(BalancedAlternativesGenerationException_wrapper));

  bp::def(
    "generate_classified_alternatives",
    &lincs::generate_classified_alternatives,
    (bp::arg("problem"), "model", "alternatives_count", "random_seed", bp::arg("max_imbalance")=std::optional<float>()),
    "Generate a set of `alternatives_count` pseudo-random alternatives for the provided `problem`, classified according to the provided `model`."
  );
  bp::def(
    "misclassify_alternatives",
    &lincs::misclassify_alternatives,
    (bp::arg("problem"), "alternatives", "count", "random_seed"),
    "Misclassify `count` alternatives from the provided `alternatives`."
  );

  bp::class_<lincs::ClassificationResult>("ClassificationResult", bp::no_init)
    .def_readonly("changed", &lincs::ClassificationResult::changed)
    .def_readonly("unchanged", &lincs::ClassificationResult::unchanged)
  ;
  bp::def(
    "classify_alternatives",
    &lincs::classify_alternatives,
    (bp::arg("problem"), "model", "alternatives"),
    "Classify the provided `alternatives` according to the provided `model`."
  );
}

}  // namespace lincs
