// Copyright 2021 Vincent Jacques

#include "dump-intermediate-models.hpp"

#include "assign.hpp"


namespace ppl {

IntermediateModelsDumper::IntermediateModelsDumper(std::ostream& stream_) :
    stream(stream_) {
  // Emitting YAML by hand... we could do better, but it works for now
  stream << "iterations:" << std::endl;
}

void IntermediateModelsDumper::after_main_iteration(int iteration_index, int, const Models<Host>& models) {
  stream
    << "  - iteration_index: " << iteration_index << "\n"
    << "    models:\n";

  auto models_view = models.get_view();

  for (uint model_index = 0; model_index != models_view.models_count; ++model_index) {
    stream
      << "      - model_index: " << model_index << "\n"
      << "        accuracy: " << get_accuracy(models, model_index) << "\n"
      << "        profiles:\n";
    for (uint profile_index = 0; profile_index != models_view.domain.categories_count - 1; ++profile_index) {
      stream << "         - [";
      for (uint crit_index = 0; crit_index != models_view.domain.criteria_count; ++crit_index) {
        if (crit_index != 0) stream << ", ";
        stream << models_view.profiles[crit_index][profile_index][model_index];
      }
      stream << "]\n";
    }
    stream
      << "        weights: [";
      for (uint crit_index = 0; crit_index != models_view.domain.criteria_count; ++crit_index) {
        if (crit_index != 0) stream << ", ";
        stream << models_view.weights[crit_index][model_index];
      }
    stream << "]\n";
  }

  stream << std::flush;
}

}  // namespace ppl
