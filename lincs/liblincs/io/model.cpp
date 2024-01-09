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

std::vector<std::vector<unsigned>> SufficientCoalitions::Roots::get_upset_roots_as_vectors() const {
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

Model::Model(const Problem& problem, const std::vector<AcceptedValues>& accepted_values_, const std::vector<SufficientCoalitions>& sufficient_coalitions_) :
  accepted_values(accepted_values_),
  sufficient_coalitions(sufficient_coalitions_)
{
  const unsigned criteria_count = problem.get_criteria().size();
  const unsigned categories_count = problem.get_ordered_categories().size();
  const unsigned boundaries_count = categories_count - 1;
  validate(accepted_values.size() == criteria_count, "The number of accepted values descriptors in the model must be equal to the number of criteria in the problem");
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    validate(accepted_values[criterion_index].get_value_type() == problem.get_criteria()[criterion_index].get_value_type(), "The value type of an accepted values descriptor must be the same as the value type of the corresponding criterion");
    dispatch(
      accepted_values[criterion_index].get(),
      [boundaries_count](const AcceptedValues::RealThresholds& thresholds) {
        validate(thresholds.get_thresholds().size() == boundaries_count, "The number of real thresholds in an accepted values descriptor must be one less than the number of categories in the problem");
      },
      [boundaries_count](const AcceptedValues::IntegerThresholds& thresholds) {
        validate(thresholds.get_thresholds().size() == boundaries_count, "The number of integer thresholds in an accepted values descriptor must be one less than the number of categories in the problem");
      },
      [boundaries_count](const AcceptedValues::EnumeratedThresholds& thresholds) {
        validate(thresholds.get_thresholds().size() == boundaries_count, "The number of enumerated thresholds in an accepted values descriptor must be one less than the number of categories in the problem");
      }
    );
  };
  validate(sufficient_coalitions.size() == boundaries_count, "The number of sufficient coalitions in the model must be one less than the number of categories in the problem");
  for (const auto& sufficient_coalitions_ : sufficient_coalitions) {
    dispatch(
      sufficient_coalitions_.get(),
      [criteria_count](const SufficientCoalitions::Weights& weights) {
        validate(weights.get_criterion_weights().size() == criteria_count, "The number of criterion weights in a sufficient coalitions descriptor must be equal to the number of criteria in the problem");
      },
      [&](const SufficientCoalitions::Roots& roots) {
        for (const auto& root: roots.get_upset_roots_as_bitsets()) {
          validate(root.size() == criteria_count, "The maximum number of elements in a root in a sufficient coalitions descriptor must be equal to the number of criteria in the problem");
        }
      }
    );
  }

  // @todo(Feature, v1.1) Validate the constraints of NCS models (inclusions of sufficient coalitions, of accepted values, etc.)
  // The issue is: we're dealing with floating point data, so we need to analyse if precision loss could lead us to reject an actually correct model.
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
  for (unsigned criterion_index = 0; criterion_index != problem.get_criteria().size(); ++criterion_index) {
    out << YAML::BeginMap;
    assert(accepted_values[criterion_index].is_thresholds());
    out << YAML::Key << "kind" << YAML::Value << "thresholds";
    out << YAML::Key << "thresholds" << YAML::Value << YAML::Flow;
    dispatch(
      accepted_values[criterion_index].get(),
      [&out](const AcceptedValues::RealThresholds& thresholds) { out << thresholds.get_thresholds(); },
      [&out](const AcceptedValues::IntegerThresholds& thresholds) { out << thresholds.get_thresholds(); },
      [&out](const AcceptedValues::EnumeratedThresholds& thresholds) { out << thresholds.get_thresholds(); }
    );
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  out << YAML::Key << "sufficient_coalitions" << YAML::Value << YAML::BeginSeq;
  for (unsigned boundary_index = 0; boundary_index != sufficient_coalitions.size(); ++boundary_index) {
    const SufficientCoalitions& sufficient_coalitions_ = sufficient_coalitions[boundary_index];
    if (use_coalitions_alias && boundary_index == 0) {
      out << YAML::Anchor("coalitions");
    }
    if (!use_coalitions_alias || boundary_index == 0) {
      out << YAML::Value << YAML::BeginMap;
      out << YAML::Key << "kind" << YAML::Value << std::string(magic_enum::enum_name(sufficient_coalitions_.get_kind()));
      dispatch(
        sufficient_coalitions_.get(),
        [&out](const SufficientCoalitions::Weights& weights) {
          out << YAML::Key << "criterion_weights" << YAML::Value << YAML::Flow << weights.get_criterion_weights();
        },
        [&out](const SufficientCoalitions::Roots& roots) {
          out << YAML::Key << "upset_roots" << YAML::Value;
          const std::vector<std::vector<unsigned>> upset_roots = roots.get_upset_roots_as_vectors();
          if (upset_roots.empty()) {
            out << YAML::Flow;
          }
          out << YAML::BeginSeq;
          for (const std::vector<unsigned>& upset_root : upset_roots) {
            out << YAML::Flow << upset_root;
          }
          out << YAML::EndSeq;
        }
      );
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
      return SufficientCoalitions(SufficientCoalitions::Weights(node["criterion_weights"].as<std::vector<float>>()));
    case SufficientCoalitions::Kind::roots:
      return SufficientCoalitions(SufficientCoalitions::Roots(problem.get_criteria().size(), node["upset_roots"].as<std::vector<std::vector<unsigned>>>()));
  }
  unreachable();
}

Model Model::load(const Problem& problem, std::istream& is) {
  CHRONE();

  const unsigned criteria_count = problem.get_criteria().size();
  const unsigned categories_count = problem.get_ordered_categories().size();
  const unsigned boundaries_count = categories_count - 1;

  YAML::Node node = YAML::Load(is);

  validator.validate(node);

  const YAML::Node& yaml_accepted_values = node["accepted_values"];
  validate(yaml_accepted_values.size() == criteria_count, "The number of accepted values descriptors in the model must be equal to the number of criteria in the problem");
  std::vector<AcceptedValues> accepted_values;
  accepted_values.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != yaml_accepted_values.size(); ++criterion_index) {
    const Criterion& criterion = problem.get_criteria()[criterion_index];
    const YAML::Node& yaml_acc_vals = yaml_accepted_values[criterion_index];
    assert(yaml_acc_vals["kind"].as<std::string>() == "thresholds");
    const YAML::Node& thresholds = yaml_acc_vals["thresholds"];

    accepted_values.push_back(dispatch(
      criterion.get_values(),
      [&thresholds, boundaries_count](const Criterion::RealValues&) {
        return AcceptedValues(AcceptedValues::RealThresholds(thresholds.as<std::vector<float>>()));
      },
      [&thresholds, boundaries_count](const Criterion::IntegerValues&) {
        return AcceptedValues(AcceptedValues::IntegerThresholds(thresholds.as<std::vector<int>>()));
      },
      [&thresholds, boundaries_count](const Criterion::EnumeratedValues&) {
        return AcceptedValues(AcceptedValues::EnumeratedThresholds(thresholds.as<std::vector<std::string>>()));
      }
    ));
  }

  const YAML::Node& yaml_sufficient_coalitions = node["sufficient_coalitions"];
  std::vector<SufficientCoalitions> sufficient_coalitions;
  sufficient_coalitions.reserve(boundaries_count);
  for (const YAML::Node& yaml_suff_coals : yaml_sufficient_coalitions) {
    sufficient_coalitions.push_back(load_sufficient_coalitions(problem, yaml_suff_coals));
  }

  return Model(problem, accepted_values, sufficient_coalitions);
}

TEST_CASE("dumping then loading model preserves data - weights") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {AcceptedValues(AcceptedValues::RealThresholds({0.4}))},
    {SufficientCoalitions(SufficientCoalitions::Weights({0.7}))},
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
      Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 2", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 3", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {
      AcceptedValues(AcceptedValues::RealThresholds({0.4})),
      AcceptedValues(AcceptedValues::RealThresholds({0.5})),
      AcceptedValues(AcceptedValues::RealThresholds({0.6})),
    },
    {SufficientCoalitions(SufficientCoalitions::Roots(3, {{0}, {1, 2}}))},
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
      Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 2", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 3", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {
      AcceptedValues(AcceptedValues::RealThresholds({0x1.259b36p-6})),
      AcceptedValues(AcceptedValues::RealThresholds({0x1.652bf4p-2})),
      AcceptedValues(AcceptedValues::RealThresholds({0x1.87662ap-3})),
    },
    {
      SufficientCoalitions(SufficientCoalitions::Weights({0x1.c78b0cp-2, 0x1.1d7974p-2, 0x1.b22782p-2})),
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
    {Criterion("Criterion 1", Criterion::IntegerValues(Criterion::PreferenceDirection::increasing, 0, 100))},
    {{"Category 1"}, {"Category 2"}, {"Category 3"}},
  };

  Model model{
    problem,
    {AcceptedValues(AcceptedValues::IntegerThresholds({40, 60}))},
    {
      SufficientCoalitions(SufficientCoalitions::Weights({0.75})),
      SufficientCoalitions(SufficientCoalitions::Weights({0.75})),
    },
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
    {Criterion("Criterion 1", Criterion::EnumeratedValues({"F", "E", "D", "C", "B", "A"}))},
    {{"Category 1"}, {"Category 2"}, {"Category 3"}},
  };

  Model model{
    problem,
    {AcceptedValues(AcceptedValues::EnumeratedThresholds({"D", "B"}))},
    {
      SufficientCoalitions(SufficientCoalitions::Weights({0.75})),
      SufficientCoalitions(SufficientCoalitions::Weights({0.75})),
    },
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
      Criterion("Criterion", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {AcceptedValues(AcceptedValues::RealThresholds({0.5}))},
    {SufficientCoalitions(SufficientCoalitions::Roots(3, {}))},
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
      Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 2", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 3", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
    },
    {{"Category 1"}, {"Category 2"}, {"Category 3"}, {"Category 4"}},
  };

  Model model{
    problem,
    {
      AcceptedValues(AcceptedValues::RealThresholds({0.2, 0.4, 0.6})),
      AcceptedValues(AcceptedValues::RealThresholds({0.3, 0.5, 0.7})),
      AcceptedValues(AcceptedValues::RealThresholds({0.4, 0.6, 0.8})),
    },
    {
      SufficientCoalitions(SufficientCoalitions::Roots(3, {{0}, {1, 2}})),
      SufficientCoalitions(SufficientCoalitions::Roots(3, {{0}, {1, 2}})),
      SufficientCoalitions(SufficientCoalitions::Roots(3, {{0}, {1, 2}})),
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
      Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 2", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 3", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
    },
    {{"Category 1"}, {"Category 2"}, {"Category 3"}, {"Category 4"}},
  };

  Model model{
    problem,
    {
      AcceptedValues(AcceptedValues::RealThresholds({0.2, 0.4, 0.6})),
      AcceptedValues(AcceptedValues::RealThresholds({0.3, 0.5, 0.7})),
      AcceptedValues(AcceptedValues::RealThresholds({0.4, 0.6, 0.8})),
    },
    {
      SufficientCoalitions(SufficientCoalitions::Roots(3, {{0}, {1, 2}})),
      SufficientCoalitions(SufficientCoalitions::Roots(3, {{1}, {0, 2}})),
      SufficientCoalitions(SufficientCoalitions::Roots(3, {{0}, {1, 2}})),
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
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
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
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
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
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
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
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
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
    "The number of accepted values descriptors in the model must be equal to the number of criteria in the problem",
    DataValidationException);
}

TEST_CASE("Validation error - size mismatch - sufficient_coalitions") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
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
    "The number of sufficient coalitions in the model must be one less than the number of categories in the problem",
    DataValidationException);
}

TEST_CASE("Validation error - size mismatch - thresholds") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
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
    "The number of real thresholds in an accepted values descriptor must be one less than the number of categories in the problem",
    DataValidationException);
}

}  // namespace lincs
