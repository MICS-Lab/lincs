#include "plad.hpp"

#include <cassert>

#include <magic_enum.hpp>
#include <yaml-cpp/yaml.h>


namespace YAML {

using plad::Domain;
using plad::Model;

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

namespace plad {

void Domain::dump(std::ostream& os) const {
  YAML::Node node;
  node["kind"] = "classification-domain";
  node["format_version"] = 1;
  node["criteria"] = criteria;
  node["categories"] = categories;

  os << node;
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

void Model::dump(std::ostream& os) const {
  YAML::Node node;
  node["kind"] = "classification-model";
  node["format_version"] = 1;
  node["boundaries"] = boundaries;

  os << node;
}

Model Model::load(Domain* domain, std::istream& is) {
  YAML::Node node = YAML::Load(is);

  assert(node["kind"].as<std::string>() == "classification-model");
  assert(node["format_version"].as<int>() == 1);

  return Model(domain, node["boundaries"].as<std::vector<Boundary>>());
}

}  // namespace plad
