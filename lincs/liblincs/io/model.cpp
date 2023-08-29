// Copyright 2023 Vincent Jacques

#include "model.hpp"

#include <cassert>

#include "../vendored/magic_enum.hpp"
#include "../vendored/yaml-cpp/yaml.h"
#include "validation.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


TEST_CASE("libyaml-cpp uses sufficient precision for floats") {
  // This test passes with libyaml-cpp version 0.7 but fails with version 0.6

  const float f = 0x1.c78b0cp-2f;

  std::stringstream ss;
  YAML::Emitter out(ss);
  out << f;

  CHECK(ss.str() == "0.444866359");

  CHECK(YAML::Load(ss).as<float>() == 0x1.c78b0cp-2f);  // No approximation: no loss of precision
}

namespace lincs {

const std::string Model::json_schema(R"($schema: https://json-schema.org/draft/2020-12/schema
title: NCS classification model
type: object
properties:
  kind:
    type: string
    const: ncs-classification-model
  format_version:
    type: integer
    const: 1
  boundaries:
    type: array
    items:
      type: object
      properties:
        profile:
          type: array
          items:
            type: number
          minItems: 1
        sufficient_coalitions:
          type: object
          oneOf:
            - properties:
                kind:
                  type: string
                  const: weights
                criterion_weights:
                  type: array
                  items:
                    type: number
                  minItems: 1
              required:
                - kind
                - criterion_weights
              additionalProperties: false
            - properties:
                kind:
                  type: string
                  const: roots
                upset_roots:
                  type: array
                  items:
                    type: array
                    items:
                      type: integer
                    minItems: 0
                  minItems: 0
              required:
                - kind
                - upset_roots
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

void Model::dump(const Problem& problem, std::ostream& os) const {
  YAML::Emitter out(os);

  bool use_coalitions_alias =
    boundaries.size() > 1
    && std::all_of(boundaries.begin(), boundaries.end(), [&](const Boundary& boundary) {
      return boundary.sufficient_coalitions == boundaries.front().sufficient_coalitions;
    });

  out << YAML::BeginMap;
  out << YAML::Key << "kind" << YAML::Value << "ncs-classification-model";
  out << YAML::Key << "format_version" << YAML::Value << 1;
  out << YAML::Key << "boundaries" << YAML::Value << YAML::BeginSeq;
  for (unsigned boundary_index = 0; boundary_index != boundaries.size(); ++boundary_index) {
    const Boundary& boundary = boundaries[boundary_index];

    out << YAML::BeginMap;
    out << YAML::Key << "profile" << YAML::Value << YAML::Flow << boundary.profile;
    out << YAML::Key << "sufficient_coalitions";
    if (use_coalitions_alias && boundary_index == 0) {
      out << YAML::Anchor("coalitions");
    }
    if (!use_coalitions_alias || boundary_index == 0) {
      out << YAML::Value << YAML::BeginMap;
      out << YAML::Key << "kind" << YAML::Value << std::string(magic_enum::enum_name(boundary.sufficient_coalitions.kind));
      switch (boundary.sufficient_coalitions.kind) {
        case SufficientCoalitions::Kind::weights:
          out << YAML::Key << "criterion_weights" << YAML::Value << YAML::Flow << boundary.sufficient_coalitions.criterion_weights;
          break;
        case SufficientCoalitions::Kind::roots:
          out << YAML::Key << "upset_roots" << YAML::Value;
          const auto upset_roots = boundary.sufficient_coalitions.upset_roots;
          if (upset_roots.empty()) {
            out << YAML::Flow;
          }
          out << YAML::BeginSeq;
          for (const auto& root : upset_roots) {
            out << YAML::Flow << YAML::BeginSeq;
            for (unsigned criterion_index = 0; criterion_index != problem.criteria.size(); ++criterion_index) {
              if (root[criterion_index]) {
                out << criterion_index;
              }
            }
            out << YAML::EndSeq;
          }
          out << YAML::EndSeq;
          break;
      }
      out << YAML::EndMap;
    } else if (use_coalitions_alias) {
      out << YAML::Value << YAML::Alias("coalitions");
    }
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;
  out << YAML::EndMap;

  os << '\n';
}

SufficientCoalitions load_sufficient_coalitions(const Problem& problem, const YAML::Node& node) {
  switch (*magic_enum::enum_cast<SufficientCoalitions::Kind>(node["kind"].as<std::string>())) {
    case SufficientCoalitions::Kind::weights:
      return SufficientCoalitions(SufficientCoalitions::weights, node["criterion_weights"].as<std::vector<float>>());
    case SufficientCoalitions::Kind::roots:
      return SufficientCoalitions(SufficientCoalitions::roots, problem.criteria.size(), node["upset_roots"].as<std::vector<std::vector<unsigned>>>());
  }
  __builtin_unreachable();
}

Model Model::load(const Problem& problem, std::istream& is) {
  YAML::Node node = YAML::Load(is);

  validator.validate(node);

  std::vector<Model::Boundary> boundaries;
  for (const YAML::Node& boundary : node["boundaries"]) {
    boundaries.emplace_back(
      boundary["profile"].as<std::vector<float>>(),
      load_sufficient_coalitions(problem, boundary["sufficient_coalitions"])
    );
  }

  return Model(problem, boundaries);
}

TEST_CASE("dumping then loading model preserves data - weights") {
  Problem problem{
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {{{0.4}, {SufficientCoalitions::weights, {0.7}}}},
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
boundaries:
  - profile: [0.400000006]
    sufficient_coalitions:
      kind: weights
      criterion_weights: [0.699999988]
)");

  Model model2 = Model::load(problem, ss);
  CHECK(model2.boundaries == model.boundaries);
}

TEST_CASE("dumping then loading model preserves data - roots") {
  Problem problem{
    {
      {"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 2", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 3", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {{{0.4, 0.5, 0.6}, {SufficientCoalitions::roots, 3, {{0}, {1, 2}}}}},
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
boundaries:
  - profile: [0.400000006, 0.5, 0.600000024]
    sufficient_coalitions:
      kind: roots
      upset_roots:
        - [0]
        - [1, 2]
)");

  Model model2 = Model::load(problem, ss);
  CHECK(model2.boundaries == model.boundaries);
}

TEST_CASE("dumping then loading model preserves data - numerical values requiring more decimal digits") {
  Problem problem{
    {
      {"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 2", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 3", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {{
      {0x1.259b36p-6, 0x1.652bf4p-2, 0x1.87662ap-3},
      {SufficientCoalitions::weights, {0x1.c78b0cp-2, 0x1.1d7974p-2, 0x1.b22782p-2}},
    }},
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
boundaries:
  - profile: [0.017920306, 0.34880048, 0.191112831]
    sufficient_coalitions:
      kind: weights
      criterion_weights: [0.444866359, 0.278783619, 0.423978835]
)");

  Model model2 = Model::load(problem, ss);
  CHECK(model2.boundaries == model.boundaries);
}

TEST_CASE("dumping empty roots uses flow style") {
  Problem problem{
    {
      {"Criterion", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {{{0.5}, {SufficientCoalitions::roots, 3, {}}}},
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
boundaries:
  - profile: [0.5]
    sufficient_coalitions:
      kind: roots
      upset_roots: []
)");

  Model model2 = Model::load(problem, ss);
  CHECK(model2.boundaries == model.boundaries);
}

TEST_CASE("dumping uses references to avoid duplication of sufficient coalitions") {
  Problem problem{
    {
      {"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 2", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 3", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
    },
    {{"Category 1"}, {"Category 2"}, {"Category 3"}, {"Category 4"}},
  };

  Model model{
    problem,
    {
      {{0.4, 0.5, 0.6}, {SufficientCoalitions::roots, 3, {{0}, {1, 2}}}},
      {{0.5, 0.6, 0.7}, {SufficientCoalitions::roots, 3, {{0}, {1, 2}}}},
      {{0.6, 0.7, 0.8}, {SufficientCoalitions::roots, 3, {{0}, {1, 2}}}},
    },
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
boundaries:
  - profile: [0.400000006, 0.5, 0.600000024]
    sufficient_coalitions: &coalitions
      kind: roots
      upset_roots:
        - [0]
        - [1, 2]
  - profile: [0.5, 0.600000024, 0.699999988]
    sufficient_coalitions: *coalitions
  - profile: [0.600000024, 0.699999988, 0.800000012]
    sufficient_coalitions: *coalitions
)");

  Model model2 = Model::load(problem, ss);
  CHECK(model2.boundaries == model.boundaries);
}

TEST_CASE("dumping doesn't use references when coalitions differ") {
  Problem problem{
    {
      {"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 2", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 3", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
    },
    {{"Category 1"}, {"Category 2"}, {"Category 3"}, {"Category 4"}},
  };

  Model model{
    problem,
    {
      {{0.4, 0.5, 0.6}, {SufficientCoalitions::roots, 3, {{0}, {1, 2}}}},
      {{0.5, 0.6, 0.7}, {SufficientCoalitions::roots, 3, {{1}, {0, 2}}}},
      {{0.6, 0.7, 0.8}, {SufficientCoalitions::roots, 3, {{0}, {1, 2}}}},
    },
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
boundaries:
  - profile: [0.400000006, 0.5, 0.600000024]
    sufficient_coalitions:
      kind: roots
      upset_roots:
        - [0]
        - [1, 2]
  - profile: [0.5, 0.600000024, 0.699999988]
    sufficient_coalitions:
      kind: roots
      upset_roots:
        - [1]
        - [0, 2]
  - profile: [0.600000024, 0.699999988, 0.800000012]
    sufficient_coalitions:
      kind: roots
      upset_roots:
        - [0]
        - [1, 2]
)");

  Model model2 = Model::load(problem, ss);
  CHECK(model2.boundaries == model.boundaries);
}

TEST_CASE("Parsing error") {
  Problem problem{
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss("*");

  CHECK_THROWS_WITH_AS(
    Model::load(problem, iss),
    "yaml-cpp: error at line 1, column 2: alias not found after *",
    YAML::Exception);
}

TEST_CASE("Validation error - not an object") {
  Problem problem{
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss("42");

  CHECK_THROWS_WITH_AS(
    Model::load(problem, iss),
    R"(JSON validation failed:
 - <root>: Value type not permitted by 'type' constraint.)",
    JsonValidationException);
}

TEST_CASE("Validation error - missing weights") {
  Problem problem{
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss(R"(kind: ncs-classification-model
format_version: 1
boundaries:
  - profile: [0.5]
    sufficient_coalitions:
      kind: weights
)");

  CHECK_THROWS_WITH_AS(
    Model::load(problem, iss),
    R"(JSON validation failed:
 - <root> [boundaries] [0] [sufficient_coalitions]: Missing required property 'criterion_weights'.
 - <root> [boundaries] [0] [sufficient_coalitions]: Failed to validate against child schema #0.
 - <root> [boundaries] [0] [sufficient_coalitions] [kind]: Failed to match expected value set by 'const' constraint.
 - <root> [boundaries] [0] [sufficient_coalitions]: Failed to validate against schema associated with property name 'kind'.
 - <root> [boundaries] [0] [sufficient_coalitions]: Missing required property 'upset_roots'.
 - <root> [boundaries] [0] [sufficient_coalitions]: Failed to validate against child schema #1.
 - <root> [boundaries] [0] [sufficient_coalitions]: Failed to validate against any child schemas allowed by oneOf constraint.
 - <root> [boundaries] [0]: Failed to validate against schema associated with property name 'sufficient_coalitions'.
 - <root> [boundaries]: Failed to validate item #0 in array.
 - <root>: Failed to validate against schema associated with property name 'boundaries'.)",
    JsonValidationException);
}

}  // namespace lincs
