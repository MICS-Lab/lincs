#include "lincs.hpp"


namespace lincs {

Model MrSortLearning::perform() {
  // @todo Implement
  Model model = Model::generate_mrsort(domain, 41);
  const unsigned criteria_count = model.boundaries[0].profile.size();
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    model.boundaries[0].profile[criterion_index] *= 0.9;
  }
  return model;
}

}  // namespace lincs
