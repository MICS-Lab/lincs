// Copyright 2023 Vincent Jacques

#include "model.hpp"

#include <cassert>

#include "../chrones.hpp"
#include "../unreachable.hpp"
#include "../vendored/magic_enum.hpp"
#include "../vendored/yaml-cpp/yaml.h"
#include "exception.hpp"
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
  accepted_values:
    description: For each criterion in the classification problem, a way to determine the accepted values for each category.
    type: array
    items:
      type: object
      oneOf:
        - properties:
            kind:
              type: string
              const: thresholds
            thresholds:
              description: For each category but the lowest, the threshold to be accepted in that category according to that criterion.
              type: array
              items:
                type: number
              minItems: 1
          required:
            - kind
            - thresholds
          additionalProperties: false
    minItems: 1
  sufficient_coalitions:
    description: For each category but the lowest, a description of the sufficient coalitions for that category.
    type: array
    items:
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
    minItems: 1
required:
  - kind
  - format_version
  - accepted_values
  - sufficient_coalitions
additionalProperties: false
)");

namespace {

std::istringstream schema_iss(Model::json_schema);
YAML::Node schema = YAML::Load(schema_iss);
JsonValidator validator(schema);

}  // namespace

std::vector<std::vector<unsigned>> SufficientCoalitions::get_upset_roots() const {
  std::vector<std::vector<unsigned>> roots;

  roots.reserve(upset_roots.size());
  for (const auto& upset_root : upset_roots) {
    std::vector<unsigned>& root = roots.emplace_back();
    for (unsigned criterion_index = 0; criterion_index != upset_root.size(); ++criterion_index) {
      if (upset_root[criterion_index]) {
        root.emplace_back(criterion_index);
      }
    }
  }

  return roots;
}

void Model::dump(const Problem& problem, std::ostream& os) const {
  CHRONE();

  #ifdef NDEBUG
  YAML::Emitter out(os);
  #else
  std::stringstream ss;
  YAML::Emitter out(ss);
  #endif

  bool use_coalitions_alias =
    boundaries.size() > 1
    && std::all_of(boundaries.begin(), boundaries.end(), [&](const Boundary& boundary) {
      return boundary.sufficient_coalitions == boundaries.front().sufficient_coalitions;
    });

  out << YAML::BeginMap;
  out << YAML::Key << "kind" << YAML::Value << "ncs-classification-model";
  out << YAML::Key << "format_version" << YAML::Value << 1;

  out << YAML::Key << "accepted_values" << YAML::Value << YAML::BeginSeq;
  for (unsigned criterion_index = 0; criterion_index != problem.criteria.size(); ++criterion_index) {
    out << YAML::BeginMap;
    out << YAML::Key << "kind" << YAML::Value << "thresholds";
    out << YAML::Key << "thresholds" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (const Boundary& boundary : boundaries) {
      out << boundary.profile[criterion_index];
    }
    out << YAML::EndSeq;
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  out << YAML::Key << "sufficient_coalitions" << YAML::Value << YAML::BeginSeq;
  for (unsigned boundary_index = 0; boundary_index != boundaries.size(); ++boundary_index) {
    const Boundary& boundary = boundaries[boundary_index];
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
          const std::vector<std::vector<unsigned>> upset_roots = boundary.sufficient_coalitions.get_upset_roots();
          if (upset_roots.empty()) {
            out << YAML::Flow;
          }
          out << YAML::BeginSeq;
          for (const std::vector<unsigned>& upset_root : upset_roots) {
            out << YAML::Flow << upset_root;
          }
          out << YAML::EndSeq;
          break;
      }
      out << YAML::EndMap;
    } else if (use_coalitions_alias) {
      out << YAML::Value << YAML::Alias("coalitions");
    }
  }
  out << YAML::EndSeq;
  out << YAML::EndMap;

  #ifndef NDEBUG
  validator.validate(YAML::Load(ss));
  os << ss.str();
  #endif

  os << '\n';
}

SufficientCoalitions load_sufficient_coalitions(const Problem& problem, const YAML::Node& node) {
  switch (*magic_enum::enum_cast<SufficientCoalitions::Kind>(node["kind"].as<std::string>())) {
    case SufficientCoalitions::Kind::weights:
      return SufficientCoalitions(SufficientCoalitions::weights, node["criterion_weights"].as<std::vector<float>>());
    case SufficientCoalitions::Kind::roots:
      return SufficientCoalitions(SufficientCoalitions::roots, problem.criteria.size(), node["upset_roots"].as<std::vector<std::vector<unsigned>>>());
  }
  unreachable();
}

Model Model::load(const Problem& problem, std::istream& is) {
  CHRONE();

  const unsigned criteria_count = problem.criteria.size();
  const unsigned categories_count = problem.categories.size();
  const unsigned boundaries_count = categories_count - 1;

  YAML::Node node = YAML::Load(is);

  validator.validate(node);

  std::vector<std::vector<float>> profiles(boundaries_count, std::vector<float>(criteria_count));
  {
    unsigned criterion_index = 0;
    const YAML::Node& accepted_values = node["accepted_values"];
    if (accepted_values.size() != criteria_count) {
      throw DataValidationException("Size mismatch: 'accepted_values' in the model file does not match the number of criteria in the problem file");
    }
    for (const YAML::Node& accepted_values : accepted_values) {
      assert(accepted_values["kind"].as<std::string>() == "thresholds");
      const std::vector<float>& thresholds = accepted_values["thresholds"].as<std::vector<float>>();
      if (thresholds.size() != boundaries_count) {
        throw DataValidationException("Size mismatch: one of 'thresholds' in the model file does not match the number of categories (minus one) in the problem file");
      }
      for (unsigned profile_index = 0; profile_index != boundaries_count; ++profile_index) {
        profiles[profile_index][criterion_index] = thresholds[profile_index];
      }
      ++criterion_index;
    }
  }

  std::reverse(profiles.begin(), profiles.end());

  const YAML::Node& sufficient_coalitions = node["sufficient_coalitions"];
  if (sufficient_coalitions.size() != boundaries_count) {
    throw DataValidationException("Size mismatch: 'sufficient_coalitions' in the model file does not match the number of categories in the problem file");
  }
  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(boundaries_count);
  for (const YAML::Node& sufficient_coalitions : sufficient_coalitions) {
    boundaries.emplace_back(
      profiles.back(),
      load_sufficient_coalitions(problem, sufficient_coalitions)
    );
    profiles.pop_back();
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
accepted_values:
  - kind: thresholds
    thresholds: [0.400000006]
sufficient_coalitions:
  - kind: weights
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
accepted_values:
  - kind: thresholds
    thresholds: [0.400000006]
  - kind: thresholds
    thresholds: [0.5]
  - kind: thresholds
    thresholds: [0.600000024]
sufficient_coalitions:
  - kind: roots
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
accepted_values:
  - kind: thresholds
    thresholds: [0.017920306]
  - kind: thresholds
    thresholds: [0.34880048]
  - kind: thresholds
    thresholds: [0.191112831]
sufficient_coalitions:
  - kind: weights
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
accepted_values:
  - kind: thresholds
    thresholds: [0.5]
sufficient_coalitions:
  - kind: roots
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
      {{0.2, 0.3, 0.4}, {SufficientCoalitions::roots, 3, {{0}, {1, 2}}}},
      {{0.4, 0.5, 0.6}, {SufficientCoalitions::roots, 3, {{0}, {1, 2}}}},
      {{0.6, 0.7, 0.8}, {SufficientCoalitions::roots, 3, {{0}, {1, 2}}}},
    },
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [0.200000003, 0.400000006, 0.600000024]
  - kind: thresholds
    thresholds: [0.300000012, 0.5, 0.699999988]
  - kind: thresholds
    thresholds: [0.400000006, 0.600000024, 0.800000012]
sufficient_coalitions:
  - &coalitions
    kind: roots
    upset_roots:
      - [0]
      - [1, 2]
  - *coalitions
  - *coalitions
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
      {{0.2, 0.3, 0.4}, {SufficientCoalitions::roots, 3, {{0}, {1, 2}}}},
      {{0.4, 0.5, 0.6}, {SufficientCoalitions::roots, 3, {{1}, {0, 2}}}},
      {{0.6, 0.7, 0.8}, {SufficientCoalitions::roots, 3, {{0}, {1, 2}}}},
    },
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [0.200000003, 0.400000006, 0.600000024]
  - kind: thresholds
    thresholds: [0.300000012, 0.5, 0.699999988]
  - kind: thresholds
    thresholds: [0.400000006, 0.600000024, 0.800000012]
sufficient_coalitions:
  - kind: roots
    upset_roots:
      - [0]
      - [1, 2]
  - kind: roots
    upset_roots:
      - [1]
      - [0, 2]
  - kind: roots
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
    DataValidationException);
}

TEST_CASE("Validation error - missing weights") {
  Problem problem{
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss(R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [0.5]
sufficient_coalitions:
  - kind: weights
)");

  CHECK_THROWS_WITH_AS(
    Model::load(problem, iss),
    R"(JSON validation failed:
 - <root> [sufficient_coalitions] [0]: Missing required property 'criterion_weights'.
 - <root> [sufficient_coalitions] [0]: Failed to validate against child schema #0.
 - <root> [sufficient_coalitions] [0] [kind]: Failed to match expected value set by 'const' constraint.
 - <root> [sufficient_coalitions] [0]: Failed to validate against schema associated with property name 'kind'.
 - <root> [sufficient_coalitions] [0]: Missing required property 'upset_roots'.
 - <root> [sufficient_coalitions] [0]: Failed to validate against child schema #1.
 - <root> [sufficient_coalitions] [0]: Failed to validate against any child schemas allowed by oneOf constraint.
 - <root> [sufficient_coalitions]: Failed to validate item #0 in array.
 - <root>: Failed to validate against schema associated with property name 'sufficient_coalitions'.)",
    DataValidationException);
}

TEST_CASE("Validation error - size mismatch - accepted_values") {
  Problem problem{
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss(R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [0.4]
  - kind: thresholds
    thresholds: [0.4]
sufficient_coalitions:
  - kind: weights
    criterion_weights: [0.7]
)");

  CHECK_THROWS_WITH_AS(
    Model::load(problem, iss),
    "Size mismatch: 'accepted_values' in the model file does not match the number of criteria in the problem file",
    DataValidationException);
}

TEST_CASE("Validation error - size mismatch - sufficient_coalitions") {
  Problem problem{
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss(R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [0.4]
sufficient_coalitions:
  - kind: weights
    criterion_weights: [0.7]
  - kind: weights
    criterion_weights: [0.7]
)");

  CHECK_THROWS_WITH_AS(
    Model::load(problem, iss),
    "Size mismatch: 'sufficient_coalitions' in the model file does not match the number of categories in the problem file",
    DataValidationException);
}

TEST_CASE("Validation error - size mismatch - thresholds") {
  Problem problem{
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss(R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [0.4, 0.5]
sufficient_coalitions:
  - kind: weights
    criterion_weights: [0.7]
)");

  CHECK_THROWS_WITH_AS(
    Model::load(problem, iss),
    "Size mismatch: one of 'thresholds' in the model file does not match the number of categories (minus one) in the problem file",
    DataValidationException);
}

}  // namespace lincs
