// Copyright 2023 Vincent Jacques

#include "model.hpp"

#include <cassert>

#include <magic_enum.hpp>
#include <yaml-cpp/yaml.h>


namespace YAML {

template<>
struct convert<lincs::Model::SufficientCoalitions> {
  static Node encode(const lincs::Model::SufficientCoalitions& coalitions) {
    Node node;
    node["kind"] = std::string(magic_enum::enum_name(coalitions.kind));
    node["criterion_weights"] = coalitions.criterion_weights;

    return node;
  }

  static bool decode(const Node& node, lincs::Model::SufficientCoalitions& coalitions) {
    coalitions.kind = magic_enum::enum_cast<lincs::Model::SufficientCoalitions::Kind>(node["kind"].as<std::string>()).value();
    coalitions.criterion_weights = node["criterion_weights"].as<std::vector<float>>();

    return true;
  }
};

template<>
struct convert<lincs::Model::Boundary> {
  static Node encode(const lincs::Model::Boundary& boundary) {
    Node node;
    node["profile"] = boundary.profile;
    node["sufficient_coalitions"] = boundary.sufficient_coalitions;

    return node;
  }

  static bool decode(const Node& node, lincs::Model::Boundary& boundary) {
    boundary.profile = node["profile"].as<std::vector<float>>();
    boundary.sufficient_coalitions = node["sufficient_coalitions"].as<lincs::Model::SufficientCoalitions>();

    return true;
  }
};

}  // namespace YAML

namespace lincs {

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

}  // namespace lincs
