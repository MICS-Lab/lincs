// Copyright 2023 Vincent Jacques

#include "problem.hpp"

#include <cassert>

#include <yaml-cpp/yaml.h>

#include "../vendored/magic_enum.hpp"
#include "validation.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace YAML {

template<>
struct convert<lincs::Category> {
  static Node encode(const lincs::Category& category) {
    Node node;
    node["name"] = category.name;

    return node;
  }

  static bool decode(const Node& node, lincs::Category& category) {
    category.name = node["name"].as<std::string>();

    return true;
  }
};

template<>
struct convert<lincs::Criterion> {
  static Node encode(const lincs::Criterion& criterion) {
    Node node;
    node["name"] = criterion.name;
    node["value_type"] = std::string(magic_enum::enum_name(criterion.value_type));
    node["category_correlation"] = std::string(magic_enum::enum_name(criterion.category_correlation));

    return node;
  }

  static bool decode(const Node& node, lincs::Criterion& criterion) {
    criterion.name = node["name"].as<std::string>();
    criterion.value_type = *magic_enum::enum_cast<lincs::Criterion::ValueType>(node["value_type"].as<std::string>());
    criterion.category_correlation = *magic_enum::enum_cast<lincs::Criterion::CategoryCorrelation>(node["category_correlation"].as<std::string>());

    return true;
  }
};

}  // namespace YAML

namespace lincs {

const std::string Problem::json_schema(R"($schema: https://json-schema.org/draft/2020-12/schema
title: Classification problem
type: object
properties:
  kind:
    type: string
    const: classification-problem
  format_version:
    type: integer
    const: 1
  criteria:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        value_type:
          type: string
          enum: [real]
        category_correlation:
          type: string
          enum: [growing]
      required:
        - name
        - value_type
        - category_correlation
      additionalProperties: false
    minItems: 1
  categories:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
      required:
        - name
      additionalProperties: false
    minItems: 2
required:
  - kind
  - format_version
  - criteria
  - categories
additionalProperties: false
)");

namespace {

std::istringstream schema_iss(Problem::json_schema);
YAML::Node schema = YAML::Load(schema_iss);
JsonValidator validator(schema);

}  // namespace

void Problem::dump(std::ostream& os) const {
  YAML::Node node;
  node["kind"] = "classification-problem";
  node["format_version"] = 1;
  node["criteria"] = criteria;
  node["categories"] = categories;

  #ifndef NDEBUG
  validator.validate(node);
  #endif

  os << node << '\n';
}

Problem Problem::load(std::istream& is) {
  YAML::Node node = YAML::Load(is);

  validator.validate(node);

  return Problem(
    node["criteria"].as<std::vector<Criterion>>(),
    node["categories"].as<std::vector<Category>>()
  );
}

TEST_CASE("dumping then loading problem preserves data") {
  Problem problem{
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing}},
    {{"Category 1"}, {"Category 2"}},
  };

  std::stringstream ss;
  problem.dump(ss);

  CHECK(ss.str() == R"(kind: classification-problem
format_version: 1
criteria:
  - name: Criterion 1
    value_type: real
    category_correlation: growing
categories:
  - name: Category 1
  - name: Category 2
)");

  Problem problem2 = Problem::load(ss);
  CHECK(problem2.criteria == problem.criteria);
  CHECK(problem2.categories == problem.categories);
}

TEST_CASE("Parsing error") {
  std::istringstream iss("*");

  CHECK_THROWS_WITH_AS(
    Problem::load(iss),
    "yaml-cpp: error at line 1, column 2: alias not found after *",
    YAML::Exception);
}

TEST_CASE("Validation error - not an object") {
  std::istringstream iss("42");

  CHECK_THROWS_WITH_AS(
    Problem::load(iss),
    "JSON validation failed:\n - <root>: Value type not permitted by 'type' constraint.",
    JsonValidationException);
}

TEST_CASE("Validation error - bad enum") {
  std::istringstream iss(R"(kind: classification-problem
format_version: 1
criteria:
  - name: Criterion 1
    value_type: invalid
    category_correlation: growing
categories:
  - name: Category 1
  - name: Category 2
)");

  CHECK_THROWS_WITH_AS(
    Problem::load(iss),
    R"(JSON validation failed:
 - <root> [criteria] [0] [value_type]: Failed to match against any enum values.
 - <root> [criteria] [0]: Failed to validate against schema associated with property name 'value_type'.
 - <root> [criteria]: Failed to validate item #0 in array.
 - <root>: Failed to validate against schema associated with property name 'criteria'.)",
    JsonValidationException);
}

}  // namespace lincs
