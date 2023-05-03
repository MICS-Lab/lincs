#include "plad.hpp"

#include <cassert>

#include <magic_enum.hpp>
#include <yaml-cpp/yaml.h>


namespace YAML {

using plad::Domain;

template<>
struct convert<Domain::Category> {
  static Node encode(const Domain::Category& category) {
    Node node;
    assert(false);
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
    assert(false);
    return node;
  }

  static bool decode(const Node& node, Domain::Criterion& criterion) {
    criterion.name = node["name"].as<std::string>();
    criterion.value_type = magic_enum::enum_cast<Domain::Criterion::ValueType>(node["value_type"].as<std::string>()).value();
    criterion.category_correlation = magic_enum::enum_cast<Domain::Criterion::CategoryCorrelation>(node["category_correlation"].as<std::string>()).value();

    return true;
  }
};

}  // namespace YAML

namespace plad {

YAML::Emitter& operator<<(YAML::Emitter& out, const Domain::Category& category) {
  out << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << category.name
      << YAML::EndMap;

  return out;
}

YAML::Emitter& operator<<(YAML::Emitter& out, const Domain::Criterion& criterion) {
  out << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << criterion.name
      << YAML::Key << "value_type" << YAML::Value << std::string(magic_enum::enum_name(criterion.value_type))
      << YAML::Key << "category_correlation" << YAML::Value << std::string(magic_enum::enum_name(criterion.category_correlation))
      << YAML::EndMap;

  return out;
}

void Domain::dump(std::ostream& os) const {
  YAML::Emitter out;

  out << YAML::BeginMap
      << YAML::Key << "criteria" << YAML::Value << criteria
      << YAML::Key << "categories" << YAML::Value << categories
      << YAML::EndMap;

  os << out.c_str();
}

Domain Domain::load(std::istream& is) {
  YAML::Node domain = YAML::Load(is);

  return Domain(
    domain["criteria"].as<std::vector<Criterion>>(),
    domain["categories"].as<std::vector<Category>>()
  );
}

}  // namespace plad
