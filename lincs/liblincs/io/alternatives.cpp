// Copyright 2023 Vincent Jacques

#include "alternatives.hpp"

#include <rapidcsv.h>


namespace lincs {

void Alternatives::dump(std::ostream& os) const {
  const unsigned criteria_count = domain.criteria.size();
  const unsigned alternatives_count = alternatives.size();

  rapidcsv::Document doc;

  doc.SetColumnName(0, "name");
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    doc.SetColumnName(criterion_index + 1, domain.criteria[criterion_index].name);
  }
  doc.SetColumnName(criteria_count + 1, "category");

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    const Alternative& alternative = alternatives[alternative_index];
    doc.SetCell<std::string>(0, alternative_index, alternative.name);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      doc.SetCell<float>(criterion_index + 1, alternative_index, alternative.profile[criterion_index]);
    }
    if (alternative.category) {
      doc.SetCell<std::string>(criteria_count + 1, alternative_index, *alternative.category);
    }
  }

  doc.Save(os);
}

Alternatives Alternatives::load(const Domain& domain, std::istream& is) {
  const unsigned criteria_count = domain.criteria.size();

  // I don't know why constructing the rapidcsv::Document directly from 'is' sometimes results in an empty document.
  // So, read the whole stream into a string and construct the document from that.
  std::string s(std::istreambuf_iterator<char>(is), {});
  std::istringstream iss(s);
  rapidcsv::Document doc(iss);

  std::vector<Alternative> alternatives;
  const unsigned alternatives_count = doc.GetRowCount();
  alternatives.reserve(alternatives_count);
  for (unsigned row_index = 0; row_index != alternatives_count; ++row_index) {
    Alternative alternative;
    alternative.name = doc.GetCell<std::string>("name", row_index);
    alternative.profile.reserve(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      alternative.profile.push_back(doc.GetCell<float>(domain.criteria[criterion_index].name, row_index));
    }
    std::string category = doc.GetCell<std::string>("category", row_index);
    if (category != "") {
      alternative.category = category;
    }
    alternatives.push_back(alternative);
  }

  return Alternatives{domain, alternatives};
}

}  // namespace lincs