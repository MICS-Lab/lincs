// Copyright 2023 Vincent Jacques

#include "model.hpp"

#include <cassert>

#include <magic_enum.hpp>
#include <yaml-cpp/yaml.h>

#include "validation.hpp"

#include <doctest.h>  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


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

const std::string Model::json_schema(R"($schema: https://json-schema.org/draft/2020-12/schema
title: NCS classification model
type: object
properties:
  kind:
    type: string
    enum: [ncs-classification-model]
  format_version:
    # type: integer  # @todo Why does this fail? (Error: <root> [format_version]: Value type not permitted by 'type' constraint.)
    enum: [1]
  boundaries:
    type: array
    items:
      type: object
      properties:
        profile:
          type: array
          # items:
          #   type: number  # @todo Why does this fail? (similar error)
          minItems: 1
        sufficient_coalitions:
          type: object
          properties:
            kind:
              type: string
              enum: [weights]
            criterion_weights:
              type: array
              # items:
              #   type: number  # @todo Why does this fail? (similar error)
              minItems: 1
          required:
            - kind
            - criterion_weights
          additionalProperties: false
      required:
        - profile
        - sufficient_coalitions
      additionalProperties: false
    minItems: 1
required:
  - kind
  - format_version
  - boundaries
additionalProperties: false
)");

namespace {

std::istringstream schema_iss(Model::json_schema);
YAML::Node schema = YAML::Load(schema_iss);
JsonValidator validator(schema);

}  // namespace

void Model::dump(std::ostream& os) const {
  YAML::Node node;
  node["kind"] = "ncs-classification-model";
  node["format_version"] = 1;
  node["boundaries"] = boundaries;

  #ifndef NDEBUG
  validator.validate(node);
  #endif

  os << node << '\n';
}

Model Model::load(const Problem& problem, std::istream& is) {
  YAML::Node node = YAML::Load(is);

  validator.validate(node);

  return Model(problem, node["boundaries"].as<std::vector<Boundary>>());
}

TEST_CASE("dumping then loading problem preserves data") {
  Problem problem{
    {{"Criterion 1", Problem::Criterion::ValueType::real, Problem::Criterion::CategoryCorrelation::growing}},
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {{{0.5}, {Model::SufficientCoalitions::Kind::weights, {1}}}},
  };

  std::stringstream ss;
  model.dump(ss);

  Model model2 = Model::load(problem, ss);
  CHECK(model2.boundaries == model.boundaries);
}

TEST_CASE("Parsing error") {
  Problem problem{
    {{"Criterion 1", Problem::Criterion::ValueType::real, Problem::Criterion::CategoryCorrelation::growing}},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss("*");

  CHECK_THROWS_WITH_AS(
    Model::load(problem, iss),
    "yaml-cpp: error at line 1, column 2: alias not found after *",
    YAML::Exception);
}

TEST_CASE("Validation error") {
  Problem problem{
    {{"Criterion 1", Problem::Criterion::ValueType::real, Problem::Criterion::CategoryCorrelation::growing}},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss("42");

  CHECK_THROWS_WITH_AS(
    Model::load(problem, iss),
    "JSON validation failed:\n - <root>: Value type not permitted by 'type' constraint.",
    JsonValidationException);
}

}  // namespace lincs
