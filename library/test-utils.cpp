// Copyright 2021-2022 Vincent Jacques

#include "test-utils.hpp"

namespace ppl {

std::shared_ptr<Domain<Host>> make_domain(
  const uint categories_count,
  const std::vector<std::pair<std::vector<float>, uint>>& alternatives_
) {
  const uint criteria_count = alternatives_.front().first.size();
  const uint alternatives_count = alternatives_.size();

  std::vector<io::ClassifiedAlternative> alternatives;
  for (auto alternative : alternatives_) {
    alternatives.push_back(io::ClassifiedAlternative(alternative.first, alternative.second));
  }

  return Domain<Host>::make(io::LearningSet(
    criteria_count, categories_count, alternatives_count, alternatives));
}

std::shared_ptr<Models<Host>> make_models(
  std::shared_ptr<Domain<Host>> domain,
  const std::vector<std::pair<std::vector<std::vector<float>>, std::vector<float>>>& models_
) {
  const uint criteria_count = domain->get_view().criteria_count;
  const uint categories_count = domain->get_view().categories_count;

  std::vector<io::Model> models;
  for (auto model : models_) {
    models.push_back(io::Model(criteria_count, categories_count, model.first, model.second));
  }

  return Models<Host>::make(domain, models);
}

}  // namespace ppl
