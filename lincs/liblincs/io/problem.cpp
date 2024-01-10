// Copyright 2023-2024 Vincent Jacques

#include "problem.hpp"

#include <cassert>

#include "../chrones.hpp"
#include "../vendored/magic_enum.hpp"
#include "../vendored/yaml-cpp/yaml.h"
#include "validation.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


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
    description: Structural information about criteria used in the classification problem.
    type: array
    items:
      type: object
      oneOf:
        - properties:
            name:
              type: string
            value_type:
              description: May be extended in the future to handle criteria with integer values, or explicitly enumerated values.
              type: string
              enum: [real, integer]
            preference_direction:
              description: May be extended in the future to handle single-peaked criteria, or criteria with unknown preference direction.
              type: string
              enum: [increasing, isotone, decreasing, antitone]
            min_value:
              type: number
            max_value:
              type: number
          required:
            - name
            - value_type
            - preference_direction
            - min_value
            - max_value
          additionalProperties: false
        - properties:
            name:
              type: string
            value_type:
              type: string
              const: enumerated
            ordered_values:
              description: Ordered list of values that can be taken by the criterion.
              type: array
              items:
                type: string
              minItems: 1
          required:
            - name
            - value_type
            - ordered_values
          additionalProperties: false
    minItems: 1
  ordered_categories:
    description: Structural information about categories in the classification problem, ordered from the worst to the best.
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
  - ordered_categories
additionalProperties: false
)");

namespace {

std::istringstream schema_iss(Problem::json_schema);
YAML::Node schema = YAML::Load(schema_iss);
JsonValidator validator(schema);

}  // namespace

void Problem::dump(std::ostream& os) const {
  CHRONE();

  #ifdef NDEBUG
  YAML::Emitter out(os);
  #else
  std::stringstream ss;
  YAML::Emitter out(ss);
  #endif

  YAML::Node node;
  out << YAML::BeginMap;
  out << YAML::Key << "kind" << YAML::Value << "classification-problem";
  out << YAML::Key << "format_version" << YAML::Value << 1;

  out << YAML::Key << "criteria" << YAML::Value << YAML::BeginSeq;
  for (const auto& criterion : criteria) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << criterion.get_name();
    out << YAML::Key << "value_type" << YAML::Value << std::string(magic_enum::enum_name(criterion.get_value_type()));
    dispatch(
      criterion.get_values(),
      [&out](const Criterion::RealValues& values) {
        out << YAML::Key << "preference_direction" << YAML::Value << std::string(magic_enum::enum_name(values.get_preference_direction()));
        out << YAML::Key << "min_value" << YAML::Value << values.get_min_value();
        out << YAML::Key << "max_value" << YAML::Value << values.get_max_value();
      },
      [&out](const Criterion::IntegerValues& values) {
        out << YAML::Key << "preference_direction" << YAML::Value << std::string(magic_enum::enum_name(values.get_preference_direction()));
        out << YAML::Key << "min_value" << YAML::Value << values.get_min_value();
        out << YAML::Key << "max_value" << YAML::Value << values.get_max_value();
      },
      [&out](const Criterion::EnumeratedValues& values) {
        out << YAML::Key << "ordered_values" << YAML::Value << YAML::Flow << values.get_ordered_values();
      }
    );
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  out << YAML::Key << "ordered_categories" << YAML::Value << YAML::BeginSeq;
  for (const auto& category : ordered_categories) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << category.get_name();
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;
  out << YAML::EndMap;

  #ifndef NDEBUG
  validator.validate(YAML::Load(ss));
  os << ss.str();
  #endif

  os << '\n';
}

Criterion::PreferenceDirection load_preference_direction(const YAML::Node& node) {
  const std::string preference_direction = node.as<std::string>();
  if (preference_direction == "isotone") {
    return Criterion::PreferenceDirection::increasing;
  } else if (preference_direction == "antitone") {
    return Criterion::PreferenceDirection::decreasing;
  } else {
    return *magic_enum::enum_cast<Criterion::PreferenceDirection>(preference_direction);
  }
}

Problem Problem::load(std::istream& is) {
  CHRONE();

  YAML::Node node = YAML::Load(is);

  validator.validate(node);

  std::vector<Criterion> criteria;
  for (const YAML::Node& criterion_node : node["criteria"]) {
    Criterion::ValueType value_type = *magic_enum::enum_cast<Criterion::ValueType>(criterion_node["value_type"].as<std::string>());
    switch (value_type) {
      case Criterion::ValueType::real:
        criteria.emplace_back(
          criterion_node["name"].as<std::string>(),
          Criterion::RealValues(
            load_preference_direction(criterion_node["preference_direction"]),
            criterion_node["min_value"].as<float>(),
            criterion_node["max_value"].as<float>()));
        break;
      case Criterion::ValueType::integer:
        criteria.emplace_back(
          criterion_node["name"].as<std::string>(),
          Criterion::IntegerValues(
            load_preference_direction(criterion_node["preference_direction"]),
            criterion_node["min_value"].as<int>(),
            criterion_node["max_value"].as<int>()));
        break;
      case Criterion::ValueType::enumerated:
        criteria.emplace_back(
          criterion_node["name"].as<std::string>(),
          Criterion::EnumeratedValues(
            criterion_node["ordered_values"].as<std::vector<std::string>>()));
        break;
    }
  }

  std::vector<Category> ordered_categories;
  for (const YAML::Node& category_node : node["ordered_categories"]) {
    ordered_categories.emplace_back(category_node["name"].as<std::string>());
  }

  return Problem(criteria, ordered_categories);
}

TEST_CASE("dumping then loading problem preserves data - real") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
    {{"Category 1"}, {"Category 2"}},
  };

  std::stringstream ss;
  problem.dump(ss);

  CHECK(ss.str() == R"(kind: classification-problem
format_version: 1
criteria:
  - name: Criterion 1
    value_type: real
    preference_direction: increasing
    min_value: 0
    max_value: 1
ordered_categories:
  - name: Category 1
  - name: Category 2
)");

  CHECK(Problem::load(ss) == problem);
}

TEST_CASE("isotone and antitone dump as increasing and decreasing") {
  Problem problem{
    {
      Criterion("Isotone criterion", Criterion::RealValues(Criterion::PreferenceDirection::isotone, 0, 1)),
      Criterion("Antitone criterion", Criterion::RealValues(Criterion::PreferenceDirection::antitone, 0, 1)),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  std::stringstream ss;
  problem.dump(ss);

  CHECK(ss.str() == R"(kind: classification-problem
format_version: 1
criteria:
  - name: Isotone criterion
    value_type: real
    preference_direction: increasing
    min_value: 0
    max_value: 1
  - name: Antitone criterion
    value_type: real
    preference_direction: decreasing
    min_value: 0
    max_value: 1
ordered_categories:
  - name: Category 1
  - name: Category 2
)");
}

TEST_CASE("isotone and antitone parse as increasing and decreasing") {
  std::istringstream iss(R"(kind: classification-problem
format_version: 1
criteria:
  - name: Isotone criterion
    value_type: real
    preference_direction: isotone
    min_value: 0
    max_value: 1
  - name: Antitone criterion
    value_type: real
    preference_direction: antitone
    min_value: 0
    max_value: 1
ordered_categories:
  - name: Category 1
  - name: Category 2
)");

  Problem problem = Problem::load(iss);

  CHECK(problem.get_criteria()[0].get_real_values().get_preference_direction() == Criterion::PreferenceDirection::isotone);
  CHECK(problem.get_criteria()[0].get_real_values().get_preference_direction() == Criterion::PreferenceDirection::increasing);
  CHECK(problem.get_criteria()[1].get_real_values().get_preference_direction() == Criterion::PreferenceDirection::antitone);
  CHECK(problem.get_criteria()[1].get_real_values().get_preference_direction() == Criterion::PreferenceDirection::decreasing);
}

TEST_CASE("dumping then loading problem preserves data - real with infinite min/max") {
  const float inf = std::numeric_limits<float>::infinity();
  Problem problem{
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::decreasing, -inf, inf))},
    {{"Category 1"}, {"Category 2"}},
  };

  std::stringstream ss;
  problem.dump(ss);

  CHECK(ss.str() == R"(kind: classification-problem
format_version: 1
criteria:
  - name: Criterion 1
    value_type: real
    preference_direction: decreasing
    min_value: -.inf
    max_value: .inf
ordered_categories:
  - name: Category 1
  - name: Category 2
)");

  CHECK(Problem::load(ss) == problem);
}

TEST_CASE("dumping then loading problem preserves data - integer") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::IntegerValues(Criterion::PreferenceDirection::increasing, 0, 20))},
    {{"Category 1"}, {"Category 2"}},
  };

  std::stringstream ss;
  problem.dump(ss);

  CHECK(ss.str() == R"(kind: classification-problem
format_version: 1
criteria:
  - name: Criterion 1
    value_type: integer
    preference_direction: increasing
    min_value: 0
    max_value: 20
ordered_categories:
  - name: Category 1
  - name: Category 2
)");

  CHECK(Problem::load(ss) == problem);
}

TEST_CASE("dumping then loading problem preserves data - enumerated") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::EnumeratedValues({"F", "E", "D", "C", "B", "A"}))},
    {{"Category 1"}, {"Category 2"}},
  };

  std::stringstream ss;
  problem.dump(ss);

  CHECK(ss.str() == R"(kind: classification-problem
format_version: 1
criteria:
  - name: Criterion 1
    value_type: enumerated
    ordered_values: [F, E, D, C, B, A]
ordered_categories:
  - name: Category 1
  - name: Category 2
)");

  CHECK(Problem::load(ss) == problem);
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
    DataValidationException);
}

TEST_CASE("Validation error - bad enum") {
  std::istringstream iss(R"(kind: classification-problem
format_version: 1
criteria:
  - name: Criterion 1
    value_type: invalid
    preference_direction: increasing
    min_value: 0
    max_value: 1
ordered_categories:
  - name: Category 1
  - name: Category 2
)");

  CHECK_THROWS_WITH_AS(
    Problem::load(iss),
    R"(JSON validation failed:
 - <root> [criteria] [0] [value_type]: Failed to match against any enum values.
 - <root> [criteria] [0]: Failed to validate against schema associated with property name 'value_type'.
 - <root> [criteria] [0]: Failed to validate against child schema #0.
 - <root> [criteria] [0] [value_type]: Failed to match expected value set by 'const' constraint.
 - <root> [criteria] [0]: Failed to validate against schema associated with property name 'value_type'.
 - <root> [criteria] [0]: Object contains a property that could not be validated using 'properties' or 'additionalProperties' constraints: 'preference_direction'.
 - <root> [criteria] [0]: Missing required property 'ordered_values'.
 - <root> [criteria] [0]: Failed to validate against child schema #1.
 - <root> [criteria] [0]: Failed to validate against any child schemas allowed by oneOf constraint.
 - <root> [criteria]: Failed to validate item #0 in array.
 - <root>: Failed to validate against schema associated with property name 'criteria'.)",
    DataValidationException);
}

}  // namespace lincs
