#include "plad.hpp"

#include <cassert>

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
    // criterion.type = Domain::ValueType::real;  // @todo Use magic_enum::enum_cast

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

// YAML::Emitter& operator<<(YAML::Emitter& out, const Domain::ValueType& value_type) {
//   // @todo Remove assert and use magic_enum.hpp to get string of name
//   assert(value_type == Domain::ValueType::real);
//   out << "real";

//   return out;
// }

YAML::Emitter& operator<<(YAML::Emitter& out, const Domain::Criterion& criterion) {
  out << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << criterion.name
      // << YAML::Key << "type" << YAML::Value << criterion.type
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
