#include "lincs.hpp"

#include <cassert>
#include <random>

#include <magic_enum.hpp>
#include <rapidcsv.h>
#include <yaml-cpp/yaml.h>


namespace YAML {

using lincs::Domain;
using lincs::Model;

template<>
struct convert<Domain::Category> {
  static Node encode(const Domain::Category& category) {
    Node node;
    node["name"] = category.name;

    return node;
  }

  static bool decode(const Node& node, Domain::Category& category) {
    category.name = node["name"].as<std::string>();

    return true;
  }
};

template<>
struct convert<Domain::Criterion> {
  static Node encode(const Domain::Criterion& criterion) {
    Node node;
    node["name"] = criterion.name;
    node["value_type"] = std::string(magic_enum::enum_name(criterion.value_type));
    node["category_correlation"] = std::string(magic_enum::enum_name(criterion.category_correlation));

    return node;
  }

  static bool decode(const Node& node, Domain::Criterion& criterion) {
    criterion.name = node["name"].as<std::string>();
    // @todo Handle error where value_type category_correlation does not properly convert back to enum
    criterion.value_type = magic_enum::enum_cast<Domain::Criterion::ValueType>(node["value_type"].as<std::string>()).value();
    criterion.category_correlation = magic_enum::enum_cast<Domain::Criterion::CategoryCorrelation>(node["category_correlation"].as<std::string>()).value();

    return true;
  }
};

template<>
struct convert<Model::SufficientCoalitions> {
  static Node encode(const Model::SufficientCoalitions& coalitions) {
    Node node;
    node["kind"] = std::string(magic_enum::enum_name(coalitions.kind));
    node["criterion_weights"] = coalitions.criterion_weights;

    return node;
  }

  static bool decode(const Node& node, Model::SufficientCoalitions& coalitions) {
    coalitions.kind = magic_enum::enum_cast<Model::SufficientCoalitions::Kind>(node["kind"].as<std::string>()).value();
    coalitions.criterion_weights = node["criterion_weights"].as<std::vector<float>>();

    return true;
  }
};

template<>
struct convert<Model::Boundary> {
  static Node encode(const Model::Boundary& boundary) {
    Node node;
    node["profile"] = boundary.profile;
    node["sufficient_coalitions"] = boundary.sufficient_coalitions;

    return node;
  }

  static bool decode(const Node& node, Model::Boundary& boundary) {
    boundary.profile = node["profile"].as<std::vector<float>>();
    boundary.sufficient_coalitions = node["sufficient_coalitions"].as<Model::SufficientCoalitions>();

    return true;
  }
};

}  // namespace YAML

namespace lincs {

void Domain::dump(std::ostream& os) const {
  YAML::Node node;
  node["kind"] = "classification-domain";
  node["format_version"] = 1;
  node["criteria"] = criteria;
  node["categories"] = categories;

  os << node << '\n';
}

Domain Domain::load(std::istream& is) {
  YAML::Node node = YAML::Load(is);

  assert(node["kind"].as<std::string>() == "classification-domain");
  assert(node["format_version"].as<int>() == 1);

  return Domain(
    node["criteria"].as<std::vector<Criterion>>(),
    node["categories"].as<std::vector<Category>>()
  );
}

Domain Domain::generate(const unsigned criteria_count, const unsigned categories_count, const unsigned random_seed) {
  // There is nothing random yet. There will be when other value types and category correlations are added.

  std::vector<Criterion> criteria;
  criteria.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    criteria.push_back(Criterion{
      "Criterion " + std::to_string(criterion_index + 1),
      Criterion::ValueType::real,
      Criterion::CategoryCorrelation::growing,
    });
  }

  std::vector<Category> categories;
  categories.reserve(categories_count);
  for (unsigned category_index = 0; category_index != categories_count; ++category_index) {
    categories.push_back(Category{
      "Category " + std::to_string(category_index + 1),
    });
  }

  return Domain{criteria, categories};
}

void Model::dump(std::ostream& os) const {
  YAML::Node node;
  node["kind"] = "classification-model";
  node["format_version"] = 1;
  node["boundaries"] = boundaries;

  os << node << '\n';
}

Model Model::load(const Domain& domain, std::istream& is) {
  YAML::Node node = YAML::Load(is);

  assert(node["kind"].as<std::string>() == "classification-model");
  assert(node["format_version"].as<int>() == 1);

  return Model(domain, node["boundaries"].as<std::vector<Boundary>>());
}

Model Model::generate_mrsort(const Domain& domain, const unsigned random_seed) {
  const unsigned categories_count = domain.categories.size();
  const unsigned criteria_count = domain.criteria.size();

  std::mt19937 gen(random_seed);

  // Profile can take any values. We arbitrarily generate them uniformly
  std::uniform_real_distribution<float> values_distribution(0.01f, 0.99f);
  // (Values clamped strictly inside ']0, 1[' to make it easier to generate balanced learning sets)
  std::vector<std::vector<float>> profiles(categories_count - 1, std::vector<float>(criteria_count));
  for (uint crit_index = 0; crit_index != criteria_count; ++crit_index) {
    // Profiles must be ordered on each criterion, so we generate a random column...
    std::vector<float> column(categories_count - 1);
    std::generate(
      column.begin(), column.end(),
      [&values_distribution, &gen]() { return values_distribution(gen); });
    // ... sort it...
    std::sort(column.begin(), column.end());
    // ... and assign that column across all profiles.
    for (uint profile_index = 0; profile_index != categories_count - 1; ++profile_index) {
      profiles[profile_index][crit_index] = column[profile_index];
    }
  }

  // Weights are a bit trickier.
  // We first generate partial sums of weights...
  std::uniform_real_distribution<float> partial_sums_distribution(0.0f, 1.f);
  std::vector<float> partial_sums(criteria_count + 1);
  partial_sums[0] = 0;  // First partial sum is zero
  std::generate(
    std::next(partial_sums.begin()), std::prev(partial_sums.end()),
    [&partial_sums_distribution, &gen]() { return partial_sums_distribution(gen); });
  partial_sums[criteria_count] = 1;  // Last partial sum is one
  // ... sort them...
  std::sort(partial_sums.begin(), partial_sums.end());
  // ... and use consecutive differences as (normalized) weights
  std::vector<float> normalized_weights(criteria_count);
  std::transform(
    partial_sums.begin(), std::prev(partial_sums.end()),
    std::next(partial_sums.begin()),
    normalized_weights.begin(),
    [](float left, float right) { return right - left; });
  // We then generate an arbitrary threshold.
  const float threshold = std::uniform_real_distribution<float>(0.0f, 1.f)(gen);
  // Finally, we denormalize weights
  std::vector<float> denormalized_weights(criteria_count);
  std::transform(
    normalized_weights.begin(), normalized_weights.end(),
    denormalized_weights.begin(),
    [threshold](float w) { return w / threshold; });

  SufficientCoalitions coalitions{
    SufficientCoalitions::Kind::weights,
    denormalized_weights,
  };

  std::vector<Boundary> boundaries;
  boundaries.reserve(categories_count - 1);
  for (unsigned category_index = 0; category_index != categories_count - 1; ++category_index) {
    boundaries.push_back(Boundary{profiles[category_index], coalitions});
  }

  return Model(domain, boundaries);
}

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

Alternatives Alternatives::generate(const Domain& domain, const Model& model, const unsigned alternatives_count, const unsigned random_seed) {
  const unsigned criteria_count = domain.criteria.size();
  std::mt19937 gen(random_seed);

  std::vector<Alternative> alternatives;
  alternatives.reserve(alternatives_count);

  // We don't do anything to ensure homogeneous repartition amongst categories.
  // We just generate random profiles uniformly in [0, 1]
  std::uniform_real_distribution<float> values_distribution(0.0f, 1.0f);

  for (uint alt_index = 0; alt_index != alternatives_count; ++alt_index) {
    std::vector<float> criteria_values(criteria_count);
    std::generate(
      criteria_values.begin(), criteria_values.end(),
      [&values_distribution, &gen]() { return values_distribution(gen); });

    alternatives.push_back(Alternative{
      "Alternative " + std::to_string(alt_index + 1),
      criteria_values,
      std::nullopt,
    });
  }

  Alternatives alts{domain, alternatives};
  classify_alternatives(domain, model, &alts);

  return alts;
}

ClassificationResult classify_alternatives(const Domain& domain, const Model& model, Alternatives* alternatives) {
  assert(&model.domain == &domain);
  assert(&alternatives.domain == &domain);

  const unsigned criteria_count = domain.criteria.size();
  const unsigned categories_count = domain.categories.size();

  ClassificationResult result{0, 0};

  for (auto& alternative: alternatives->alternatives) {
    uint category_index;
    for (category_index = categories_count - 1; category_index != 0; --category_index) {
      const auto& boundary = model.boundaries[category_index - 1];
      assert(boundary.sufficient_coalitions.kind == SufficientCoalitions::Kind::weights);
      float weight_at_or_above_profile = 0;
      for (uint criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        const float alternative_value = alternative.profile[criterion_index];
        const float profile_value = boundary.profile[criterion_index];
        if (alternative_value >= profile_value) {
          weight_at_or_above_profile += boundary.sufficient_coalitions.criterion_weights[criterion_index];
        }
      }
      if (weight_at_or_above_profile >= 1.f) {
        break;
      }
    }

    const std::string& category = domain.categories[category_index].name;
    if (alternative.category == category) {
      ++result.unchanged;
    } else {
      alternative.category = category;
      ++result.changed;
    }
  }

  return result;
}

}  // namespace lincs
