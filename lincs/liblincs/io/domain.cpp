// Copyright 2023 Vincent Jacques

#include "domain.hpp"

#include <cassert>

#include <magic_enum.hpp>
#include <yaml-cpp/yaml.h>

#include <doctest.h>  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace YAML {

template<>
struct convert<lincs::Domain::Category> {
  static Node encode(const lincs::Domain::Category& category) {
    Node node;
    node["name"] = category.name;

    return node;
  }

  static bool decode(const Node& node, lincs::Domain::Category& category) {
    category.name = node["name"].as<std::string>();

    return true;
  }
};

template<>
struct convert<lincs::Domain::Criterion> {
  static Node encode(const lincs::Domain::Criterion& criterion) {
    Node node;
    node["name"] = criterion.name;
    node["value_type"] = std::string(magic_enum::enum_name(criterion.value_type));
    node["category_correlation"] = std::string(magic_enum::enum_name(criterion.category_correlation));

    return node;
  }

  static bool decode(const Node& node, lincs::Domain::Criterion& criterion) {
    criterion.name = node["name"].as<std::string>();
    // @todo Handle error where value_type category_correlation does not properly convert back to enum
    criterion.value_type = magic_enum::enum_cast<lincs::Domain::Criterion::ValueType>(node["value_type"].as<std::string>()).value();
    criterion.category_correlation = magic_enum::enum_cast<lincs::Domain::Criterion::CategoryCorrelation>(node["category_correlation"].as<std::string>()).value();

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

TEST_CASE("dumping then loading domain preserves data") {
  Domain domain{
    {{"Criterion 1", Domain::Criterion::ValueType::real, Domain::Criterion::CategoryCorrelation::growing}},
    {{"Category 1"}, {"Category 2"}},
  };

  std::stringstream ss;
  domain.dump(ss);

  Domain domain2 = Domain::load(ss);
  CHECK(domain2.criteria == domain.criteria);
  CHECK(domain2.categories == domain.categories);
}

}  // namespace lincs
