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
struct convert<Domain> {
  static Node encode(const Domain& domain) {
    Node node;
    node["kind"] = "classification-domain";
    node["format_version"] = 1;
    node["criteria"] = domain.criteria;
    node["categories"] = domain.categories;

    return node;
  }

  static bool decode(const Node& node, Domain& domain) {
    assert(node["kind"].as<std::string>() == "classification-domain");
    assert(node["format_version"].as<int>() == 1);

    domain.criteria = node["criteria"].as<std::vector<Domain::Criterion>>();
    domain.categories = node["categories"].as<std::vector<Domain::Category>>();

    return true;
  }
};

}  // namespace YAML

namespace plad {

void Domain::dump(std::ostream& os) const {
  os << YAML::Node(*this);
}

Domain Domain::load(std::istream& is) {
  return YAML::Load(is).as<Domain>();
}

}  // namespace plad
