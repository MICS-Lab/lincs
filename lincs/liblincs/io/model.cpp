// Copyright 2023 Vincent Jacques

#include "model.hpp"

#include <cassert>

#include "../chrones.hpp"
#include "../unreachable.hpp"
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

std::vector<std::vector<unsigned>> SufficientCoalitions::get_upset_roots_as_vectors() const {
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
    sufficient_coalitions.size() > 1
    && std::all_of(std::next(sufficient_coalitions.begin()), sufficient_coalitions.end(), [&](const SufficientCoalitions& suff_coals) {
      return suff_coals == sufficient_coalitions.front();
    });

  out << YAML::BeginMap;
  out << YAML::Key << "kind" << YAML::Value << "ncs-classification-model";
  out << YAML::Key << "format_version" << YAML::Value << 1;

  out << YAML::Key << "accepted_values" << YAML::Value << YAML::BeginSeq;
  for (unsigned criterion_index = 0; criterion_index != problem.criteria.size(); ++criterion_index) {
    const Criterion& criterion = problem.criteria[criterion_index];
    const AcceptedValues& acc_vals = accepted_values[criterion_index];
    out << YAML::BeginMap;
    out << YAML::Key << "kind" << YAML::Value << "thresholds";
    out << YAML::Key << "thresholds" << YAML::Value << YAML::Flow;
    switch (criterion.get_value_type()) {
      case Criterion::ValueType::real:
        out << acc_vals.get_real_thresholds();
        break;
      case Criterion::ValueType::integer:
        out << acc_vals.get_integer_thresholds();
        break;
      case Criterion::ValueType::enumerated:
        out << acc_vals.get_enumerated_thresholds();
        break;
    }
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  out << YAML::Key << "sufficient_coalitions" << YAML::Value << YAML::BeginSeq;
  for (unsigned boundary_index = 0; boundary_index != sufficient_coalitions.size(); ++boundary_index) {
    const SufficientCoalitions& suff_coals = sufficient_coalitions[boundary_index];
    if (use_coalitions_alias && boundary_index == 0) {
      out << YAML::Anchor("coalitions");
    }
    if (!use_coalitions_alias || boundary_index == 0) {
      out << YAML::Value << YAML::BeginMap;
      out << YAML::Key << "kind" << YAML::Value << std::string(magic_enum::enum_name(suff_coals.get_kind()));
      switch (suff_coals.get_kind()) {
        case SufficientCoalitions::Kind::weights:
          out << YAML::Key << "criterion_weights" << YAML::Value << YAML::Flow << suff_coals.get_criterion_weights();
          break;
        case SufficientCoalitions::Kind::roots:
          out << YAML::Key << "upset_roots" << YAML::Value;
          const std::vector<std::vector<unsigned>> upset_roots = suff_coals.get_upset_roots_as_vectors();
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
      return SufficientCoalitions::make_weights(node["criterion_weights"].as<std::vector<float>>());
    case SufficientCoalitions::Kind::roots:
      return SufficientCoalitions::make_roots_from_vectors(problem.criteria.size(), node["upset_roots"].as<std::vector<std::vector<unsigned>>>());
  }
  unreachable();
}

Model Model::load(const Problem& problem, std::istream& is) {
  CHRONE();

  const unsigned criteria_count = problem.criteria.size();
  const unsigned categories_count = problem.ordered_categories.size();
  const unsigned boundaries_count = categories_count - 1;

  YAML::Node node = YAML::Load(is);

  validator.validate(node);

  const YAML::Node& yaml_accepted_values = node["accepted_values"];
  if (yaml_accepted_values.size() != criteria_count) {
    throw DataValidationException("Size mismatch: 'accepted_values' in the model file does not match the number of criteria in the problem file");
  }
  std::vector<AcceptedValues> accepted_values;
  accepted_values.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    const Criterion& criterion = problem.criteria[criterion_index];
    const YAML::Node& yaml_acc_vals = yaml_accepted_values[criterion_index];
    assert(yaml_acc_vals["kind"].as<std::string>() == "thresholds");

    switch (criterion.get_value_type()) {
      case Criterion::ValueType::real:
        {
          const std::vector<float> thresholds = yaml_acc_vals["thresholds"].as<std::vector<float>>();
          if (thresholds.size() != boundaries_count) {
            throw DataValidationException("Size mismatch: one of 'thresholds' in the model file does not match the number of categories (minus one) in the problem file");
          }
          accepted_values.push_back(AcceptedValues::make_real_thresholds(thresholds));
        }
        break;
      case Criterion::ValueType::integer:
        {
          const std::vector<int> thresholds = yaml_acc_vals["thresholds"].as<std::vector<int>>();
          if (thresholds.size() != boundaries_count) {
            throw DataValidationException("Size mismatch: one of 'thresholds' in the model file does not match the number of categories (minus one) in the problem file");
          }
          accepted_values.push_back(AcceptedValues::make_integer_thresholds(thresholds));
        }
        break;
      case Criterion::ValueType::enumerated:
        {
          const std::vector<std::string> thresholds = yaml_acc_vals["thresholds"].as<std::vector<std::string>>();
          if (thresholds.size() != boundaries_count) {
            throw DataValidationException("Size mismatch: one of 'thresholds' in the model file does not match the number of categories (minus one) in the problem file");
          }
          accepted_values.push_back(AcceptedValues::make_enumerated_thresholds(thresholds));
        }
        break;
    }
  }

  const YAML::Node& yaml_sufficient_coalitions = node["sufficient_coalitions"];
  if (yaml_sufficient_coalitions.size() != boundaries_count) {
    throw DataValidationException("Size mismatch: 'sufficient_coalitions' in the model file does not match the number of categories in the problem file");
  }
  std::vector<SufficientCoalitions> sufficient_coalitions;
  sufficient_coalitions.reserve(boundaries_count);
  for (const YAML::Node& yaml_suff_coals : yaml_sufficient_coalitions) {
    sufficient_coalitions.push_back(load_sufficient_coalitions(problem, yaml_suff_coals));
  }

  return Model(problem, accepted_values, sufficient_coalitions);
}

TEST_CASE("dumping then loading model preserves data - weights") {
  Problem problem{
    {Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1)},
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {AcceptedValues::make_real_thresholds({0.4})},
    {SufficientCoalitions::make_weights({0.7})},
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

  CHECK(Model::load(problem, ss) == model);
}

TEST_CASE("dumping then loading model preserves data - roots") {
  Problem problem{
    {
      Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 2", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 3", Criterion::PreferenceDirection::increasing, 0, 1),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {
      AcceptedValues::make_real_thresholds({0.4}),
      AcceptedValues::make_real_thresholds({0.5}),
      AcceptedValues::make_real_thresholds({0.6}),
    },
    {SufficientCoalitions::make_roots_from_vectors(3, {{0}, {1, 2}})},
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

  CHECK(Model::load(problem, ss) == model);
}

TEST_CASE("dumping then loading model preserves data - numerical values requiring more decimal digits") {
  Problem problem{
    {
      Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 2", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 3", Criterion::PreferenceDirection::increasing, 0, 1),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {
      AcceptedValues::make_real_thresholds({0x1.259b36p-6}),
      AcceptedValues::make_real_thresholds({0x1.652bf4p-2}),
      AcceptedValues::make_real_thresholds({0x1.87662ap-3}),
    },
    {
      SufficientCoalitions::make_weights({0x1.c78b0cp-2, 0x1.1d7974p-2, 0x1.b22782p-2}),
    },
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

  CHECK(Model::load(problem, ss) == model);
}

TEST_CASE("dumping then loading model preserves data - integer criterion") {
  Problem problem{
    {Criterion::make_integer("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 100)},
    {{"Category 1"}, {"Category 2"}, {"Category 3"}},
  };

  Model model{
    problem,
    {AcceptedValues::make_integer_thresholds({40, 60})},
    {SufficientCoalitions::make_weights({0.75}), SufficientCoalitions::make_weights({0.75})},
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [40, 60]
sufficient_coalitions:
  - &coalitions
    kind: weights
    criterion_weights: [0.75]
  - *coalitions
)");

  CHECK(Model::load(problem, ss) == model);
}

TEST_CASE("dumping then loading model preserves data - enumerated criterion") {
  Problem problem{
    {Criterion::make_enumerated("Criterion 1", {"F", "E", "D", "C", "B", "A"})},
    {{"Category 1"}, {"Category 2"}, {"Category 3"}},
  };

  Model model{
    problem,
    {AcceptedValues::make_enumerated_thresholds({"D", "B"})},
    {SufficientCoalitions::make_weights({0.75}), SufficientCoalitions::make_weights({0.75})},
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [D, B]
sufficient_coalitions:
  - &coalitions
    kind: weights
    criterion_weights: [0.75]
  - *coalitions
)");

  CHECK(Model::load(problem, ss) == model);
}

TEST_CASE("dumping empty roots uses flow style") {
  Problem problem{
    {
      Criterion::make_real("Criterion", Criterion::PreferenceDirection::increasing, 0, 1),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {AcceptedValues::make_real_thresholds({0.5})},
    {SufficientCoalitions::make_roots_from_vectors(3, {})},
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

  CHECK(Model::load(problem, ss) == model);
}

TEST_CASE("dumping uses references to avoid duplication of sufficient coalitions") {
  Problem problem{
    {
      Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 2", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 3", Criterion::PreferenceDirection::increasing, 0, 1),
    },
    {{"Category 1"}, {"Category 2"}, {"Category 3"}, {"Category 4"}},
  };

  Model model{
    problem,
    {
      AcceptedValues::make_real_thresholds({0.2, 0.4, 0.6}),
      AcceptedValues::make_real_thresholds({0.3, 0.5, 0.7}),
      AcceptedValues::make_real_thresholds({0.4, 0.6, 0.8}),
    },
    {
      SufficientCoalitions::make_roots_from_vectors(3, {{0}, {1, 2}}),
      SufficientCoalitions::make_roots_from_vectors(3, {{0}, {1, 2}}),
      SufficientCoalitions::make_roots_from_vectors(3, {{0}, {1, 2}}),
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

  CHECK(Model::load(problem, ss) == model);
}

TEST_CASE("dumping doesn't use references when coalitions differ") {
  Problem problem{
    {
      Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 2", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 3", Criterion::PreferenceDirection::increasing, 0, 1),
    },
    {{"Category 1"}, {"Category 2"}, {"Category 3"}, {"Category 4"}},
  };

  Model model{
    problem,
    {
      AcceptedValues::make_real_thresholds({0.2, 0.4, 0.6}),
      AcceptedValues::make_real_thresholds({0.3, 0.5, 0.7}),
      AcceptedValues::make_real_thresholds({0.4, 0.6, 0.8}),
    },
    {
      SufficientCoalitions::make_roots_from_vectors(3, {{0}, {1, 2}}),
      SufficientCoalitions::make_roots_from_vectors(3, {{1}, {0, 2}}),
      SufficientCoalitions::make_roots_from_vectors(3, {{0}, {1, 2}}),
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

  CHECK(Model::load(problem, ss) == model);
}

TEST_CASE("Parsing error") {
  Problem problem{
    {Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1)},
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
    {Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1)},
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
    {Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1)},
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
    {Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1)},
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
    {Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1)},
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
    {Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1)},
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
