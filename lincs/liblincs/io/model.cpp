// Copyright 2023-2024 Vincent Jacques

#include "model.hpp"

#include <cassert>

#include "../chrones.hpp"
#include "../classification.hpp"
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

namespace YAML {

template<typename T>
Emitter& operator<<(Emitter& out, const std::optional<T>& o) {
  if (o) {
    out << *o;
  } else {
    out << Null;
  }
  return out;
}

template <typename T>
struct convert<std::optional<T>> {
  static bool decode(const Node& node, std::optional<T>& rhs) {
    if (node.IsNull()) {
      rhs.reset();
    } else {
      rhs = node.as<T>();
    }
    return true;
  }
};

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
        - properties:
            kind:
              type: string
              const: intervals
            intervals:
              description: For each category but the lowest, the interval of values to be accepted in that category according to that criterion.
              type: array
              minItems: 1
              items:
                oneOf:
                  - type: 'null'
                  - type: array
                    minItems: 2
                    maxItems: 2
                    items:
                      type: number
          required:
            - kind
            - intervals
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
    const auto& criterion = problem.get_criteria()[criterion_index];
    validate(accepted_values[criterion_index].get_value_type() == criterion.get_value_type(), "The value type of an accepted values descriptor must be the same as the value type of the corresponding criterion");
    dispatch(
      accepted_values[criterion_index].get(),
      [&criterion, criterion_index, boundaries_count](const AcceptedValues::RealThresholds& thresholds) {
        validate(thresholds.get_thresholds().size() == boundaries_count, "The number of real thresholds in an accepted values descriptor must be one less than the number of categories in the problem");
        const auto& criterion_values = criterion.get_real_values();
        for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
          const std::optional<float> threshold = thresholds.get_thresholds()[boundary_index];
          if (threshold) {
            validate(criterion_values.is_acceptable(*threshold), "Each threshold in an accepted values descriptor must be between the min and max values for the corresponding real criterion");
          }
        }
        for (unsigned boundary_index = 1; boundary_index != boundaries_count; ++boundary_index) {
          const std::optional<float> previous_threshold = thresholds.get_thresholds()[boundary_index - 1];
          const std::optional<float> threshold = thresholds.get_thresholds()[boundary_index];
          if (previous_threshold) {
            if (threshold) {
              switch (criterion_values.get_preference_direction()) {
                case Criterion::PreferenceDirection::increasing:
                  validate(*threshold >= *previous_threshold, "The real thresholds in an accepted values descriptor must be in preference order");
                  break;
                case Criterion::PreferenceDirection::decreasing:
                  validate(*threshold <= *previous_threshold, "The real thresholds in an accepted values descriptor must be in preference order");
                  break;
                default:
                  validate(false, "Thresholds accepted values descriptors are only supported for monotonic criteria");
                  break;
              }
            }
          } else {
            validate(!threshold, "After a null threshold, all subsequent thresholds must be null");
          }
        }
      },
      [&criterion, criterion_index, boundaries_count](const AcceptedValues::IntegerThresholds& thresholds) {
        validate(thresholds.get_thresholds().size() == boundaries_count, "The number of integer thresholds in an accepted values descriptor must be one less than the number of categories in the problem");
        const auto& criterion_values = criterion.get_integer_values();
        for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
          const std::optional<int> threshold = thresholds.get_thresholds()[boundary_index];
          if (threshold) {
            validate(criterion_values.is_acceptable(*threshold), "Each threshold in an accepted values descriptor must be between the min and max values for the corresponding integer criterion");
          }
        }
        for (unsigned boundary_index = 1; boundary_index != boundaries_count; ++boundary_index) {
          const std::optional<int> previous_threshold = thresholds.get_thresholds()[boundary_index - 1];
          const std::optional<int> threshold = thresholds.get_thresholds()[boundary_index];
          if (previous_threshold) {
            if (threshold) {
              switch (criterion_values.get_preference_direction()) {
                case Criterion::PreferenceDirection::increasing:
                  validate(*threshold >= *previous_threshold, "The integer thresholds in an accepted values descriptor must be in preference order");
                  break;
                case Criterion::PreferenceDirection::decreasing:
                  validate(*threshold <= *previous_threshold, "The integer thresholds in an accepted values descriptor must be in preference order");
                  break;
                default:
                  validate(false, "Thresholds accepted values descriptors are only supported for monotonic criteria");
                  break;
              }
            }
          } else {
            validate(!threshold, "After a null threshold, all subsequent thresholds must be null");
          }
        }
      },
      [&criterion, criterion_index, boundaries_count](const AcceptedValues::EnumeratedThresholds& thresholds) {
        validate(thresholds.get_thresholds().size() == boundaries_count, "The number of enumerated thresholds in an accepted values descriptor must be one less than the number of categories in the problem");
        const auto& criterion_values = criterion.get_enumerated_values();
        for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
          const std::optional<std::string>& threshold = thresholds.get_thresholds()[boundary_index];
          if (threshold) {
            validate(criterion_values.is_acceptable(*threshold), "Each threshold in an accepted values descriptor must be in the enumerated values for the corresponding criterion");
          }
        }
        for (unsigned boundary_index = 1; boundary_index != boundaries_count; ++boundary_index) {
          const std::optional<std::string>& previous_threshold = thresholds.get_thresholds()[boundary_index - 1];
          const std::optional<std::string>& threshold = thresholds.get_thresholds()[boundary_index];
          if (previous_threshold) {
            if (threshold) {
              validate(
                criterion_values.get_value_rank(*threshold) >= criterion_values.get_value_rank(*previous_threshold),
                "The enumerated thresholds in an accepted values descriptor must be in preference order"
              );
            }
          } else {
            validate(!threshold, "After a null threshold, all subsequent thresholds must be null");
          }
        }
      },
      [&criterion, criterion_index, boundaries_count](const AcceptedValues::RealIntervals& intervals) {
        validate(intervals.get_intervals().size() == boundaries_count, "The number of real intervals in an accepted values descriptor must be one less than the number of categories in the problem");
        const auto& criterion_values = criterion.get_real_values();
        for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
          const auto interval = intervals.get_intervals()[boundary_index];
          if (interval) {
            validate(
              criterion_values.is_acceptable(interval->first) && criterion_values.is_acceptable(interval->second),
              "Both ends of each interval in an accepted values descriptor must be between the min and max values for the corresponding real criterion");
          }
        }
        validate(criterion_values.get_preference_direction() == Criterion::PreferenceDirection::single_peaked, "Intervals accepted values descriptors are only supported for single-peaked criteria");
        for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
          const auto interval = intervals.get_intervals()[boundary_index];
          if (interval) {
            validate(
              interval->first <= interval->second,
              "The ends of intervals accepted values descriptors for real criteria  must be in order");
          }
        }
        for (unsigned boundary_index = 1; boundary_index != boundaries_count; ++boundary_index) {
          const auto previous_interval = intervals.get_intervals()[boundary_index - 1];
          const auto interval = intervals.get_intervals()[boundary_index];
          if (previous_interval) {
            if (interval) {
              validate(
                previous_interval->first <= interval->first
                && previous_interval->second >= interval->second,
                "Intervals accepted values descriptors for real criteria must be nested");
            }
          } else {
            validate(!interval, "After a null interval, all subsequent intervals must be null");
          }
        }
      },
      [&criterion, criterion_index, boundaries_count](const AcceptedValues::IntegerIntervals& intervals) {
        validate(intervals.get_intervals().size() == boundaries_count, "The number of integer intervals in an accepted values descriptor must be one less than the number of categories in the problem");
        const auto& criterion_values = criterion.get_integer_values();
        for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
          const auto interval = intervals.get_intervals()[boundary_index];
          if (interval) {
            validate(
              criterion_values.is_acceptable(interval->first) && criterion_values.is_acceptable(interval->second),
              "Both ends of each interval in an accepted values descriptor must be between the min and max values for the corresponding integer criterion");
          }
        }
        validate(criterion_values.get_preference_direction() == Criterion::PreferenceDirection::single_peaked, "Intervals accepted values descriptors are only supported for single-peaked criteria");
        for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
          const auto interval = intervals.get_intervals()[boundary_index];
          if (interval) {
            validate(
              interval->first <= interval->second,
              "The ends of intervals accepted values descriptors for integer criteria  must be in order");
          }
        }
        for (unsigned boundary_index = 1; boundary_index != boundaries_count; ++boundary_index) {
          const auto previous_interval = intervals.get_intervals()[boundary_index - 1];
          const auto interval = intervals.get_intervals()[boundary_index];
          if (previous_interval) {
            if (interval) {
              validate(
                previous_interval->first <= interval->first
                && previous_interval->second >= interval->second,
                "Intervals accepted values descriptors for integer criteria must be nested");
            }
          } else {
            validate(!interval, "After a null interval, all subsequent intervals must be null");
          }
        }
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
  for (unsigned coalition_index = 0; coalition_index != unsigned(1 << criteria_count); ++coalition_index) {
    boost::dynamic_bitset<> coalition(criteria_count, coalition_index);
    for (unsigned boundary_index = 1; boundary_index != boundaries_count; ++boundary_index) {
      bool accepted_by_upper = std::visit(
        [&coalition](const auto& sufficient_coalitions_) {
          return sufficient_coalitions_.accept(coalition);
        },
        sufficient_coalitions[boundary_index].get()
      );
      if (accepted_by_upper) {
        const bool accepted_by_lower = std::visit(
          [&coalition](const auto& sufficient_coalitions_) {
            return sufficient_coalitions_.accept(coalition);
          },
          sufficient_coalitions[boundary_index - 1].get()
        );
        validate(accepted_by_lower, "Sufficient coalitions must be imbricated");
      }
    }
  }
}

void Model::dump(const Problem& problem, std::ostream& os) const {
  CHRONE();

  #ifdef NDEBUG
  YAML::Emitter out(os);
  #else
  std::stringstream ss;
  YAML::Emitter out(ss);
  #endif

  out.SetNullFormat(YAML::EMITTER_MANIP::LowerNull);

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
    dispatch(
      accepted_values[criterion_index].get(),
      [&out](const AcceptedValues::RealThresholds& thresholds) {
        out << YAML::Key << "kind" << YAML::Value << "thresholds";
        out << YAML::Key << "thresholds" << YAML::Value << YAML::Flow;
        out << thresholds.get_thresholds();
      },
      [&out](const AcceptedValues::IntegerThresholds& thresholds) {
        out << YAML::Key << "kind" << YAML::Value << "thresholds";
        out << YAML::Key << "thresholds" << YAML::Value << YAML::Flow;
        out << thresholds.get_thresholds();
      },
      [&out](const AcceptedValues::EnumeratedThresholds& thresholds) {
        out << YAML::Key << "kind" << YAML::Value << "thresholds";
        out << YAML::Key << "thresholds" << YAML::Value << YAML::Flow;
        out << thresholds.get_thresholds();
      },
      [&out](const AcceptedValues::RealIntervals& intervals) {
        out << YAML::Key << "kind" << YAML::Value << "intervals";
        out << YAML::Key << "intervals" << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (const auto& interval : intervals.get_intervals()) {
          if (interval) {
            out << YAML::BeginSeq << interval->first << interval->second << YAML::EndSeq;
          } else {
            out << YAML::Null;
          }
        }
        out << YAML::EndSeq;
      },
      [&out](const AcceptedValues::IntegerIntervals& intervals) {
        out << YAML::Key << "kind" << YAML::Value << "intervals";
        out << YAML::Key << "intervals" << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (const auto& interval : intervals.get_intervals()) {
          if (interval) {
            out << YAML::BeginSeq << interval->first << interval->second << YAML::EndSeq;
          } else {
            out << YAML::Null;
          }
        }
        out << YAML::EndSeq;
      }
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
      return SufficientCoalitions(SufficientCoalitions::Roots(problem, node["upset_roots"].as<std::vector<std::vector<unsigned>>>()));
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
    if (yaml_acc_vals["kind"].as<std::string>() == "thresholds") {
      const YAML::Node& thresholds = yaml_acc_vals["thresholds"];

      accepted_values.push_back(dispatch(
        criterion.get_values(),
        [&thresholds, boundaries_count](const Criterion::RealValues&) {
          return AcceptedValues(AcceptedValues::RealThresholds(thresholds.as<std::vector<std::optional<float>>>()));
        },
        [&thresholds, boundaries_count](const Criterion::IntegerValues&) {
          return AcceptedValues(AcceptedValues::IntegerThresholds(thresholds.as<std::vector<std::optional<int>>>()));
        },
        [&thresholds, boundaries_count](const Criterion::EnumeratedValues&) {
          return AcceptedValues(AcceptedValues::EnumeratedThresholds(thresholds.as<std::vector<std::optional<std::string>>>()));
        }
      ));
    } else {
      assert(yaml_acc_vals["kind"].as<std::string>() == "intervals");

      const YAML::Node& intervals = yaml_acc_vals["intervals"];
      accepted_values.push_back(dispatch(
        criterion.get_values(),
        [&intervals, boundaries_count](const Criterion::RealValues&) {
          return AcceptedValues(AcceptedValues::RealIntervals(intervals.as<std::vector<std::optional<std::pair<float, float>>>>()));
        },
        [&intervals, boundaries_count](const Criterion::IntegerValues&) {
          return AcceptedValues(AcceptedValues::IntegerIntervals(intervals.as<std::vector<std::optional<std::pair<int, int>>>>()));
        },
        [](const Criterion::EnumeratedValues&) -> AcceptedValues {
          unreachable();
        }
      ));
    }
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
    {SufficientCoalitions(SufficientCoalitions::Roots(problem, {{0}, {1, 2}}))},
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

TEST_CASE("dumping then loading model preserves data - null thresholds") {
  Problem problem{
    {
      Criterion("Real", Criterion::RealValues(Criterion::PreferenceDirection::increasing, -1, 1)),
      Criterion("Integer", Criterion::IntegerValues(Criterion::PreferenceDirection::increasing, 0, 100)),
    },
    {{"Cat 1"}, {"Cat 2"}, {"Cat 3"}, {"Cat 4"}, {"Cat 5"}},
  };

  Model model{
    problem,
    {
      AcceptedValues(AcceptedValues::RealThresholds({-0.5, 0, std::nullopt, std::nullopt})),
      AcceptedValues(AcceptedValues::IntegerThresholds({20, 40, 60, std::nullopt})),
    },
    {
      SufficientCoalitions(SufficientCoalitions::Weights({0.5, 0.5})),
      SufficientCoalitions(SufficientCoalitions::Weights({0.5, 0.5})),
      SufficientCoalitions(SufficientCoalitions::Weights({0.5, 0.5})),
      SufficientCoalitions(SufficientCoalitions::Weights({0.5, 0.5})),
    },
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [-0.5, 0, null, null]
  - kind: thresholds
    thresholds: [20, 40, 60, null]
sufficient_coalitions:
  - &coalitions
    kind: weights
    criterion_weights: [0.5, 0.5]
  - *coalitions
  - *coalitions
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

TEST_CASE("dumping then loading model preserves data - single-peaked criteria") {
  Problem problem{
    {
      Criterion("Real", Criterion::RealValues(Criterion::PreferenceDirection::single_peaked, -10, 10)),
      Criterion("Integer", Criterion::IntegerValues(Criterion::PreferenceDirection::single_peaked, 0, 100)),
    },
    {{"Cat 1"}, {"Cat 2"}, {"Cat 3"}, {"Cat 4"}},
  };

  Model model{
    problem,
    {
      AcceptedValues(AcceptedValues::RealIntervals({std::make_pair(-8.5, 8.5), std::make_pair(-3.5, 5.5), std::nullopt})),
      AcceptedValues(AcceptedValues::IntegerIntervals({std::make_pair(20, 80), std::make_pair(40, 60), std::nullopt})),
    },
    {
      SufficientCoalitions(SufficientCoalitions::Weights({0.5, 0.5})),
      SufficientCoalitions(SufficientCoalitions::Weights({0.5, 0.5})),
      SufficientCoalitions(SufficientCoalitions::Weights({0.5, 0.5})),
    },
  };

  std::stringstream ss;
  model.dump(problem, ss);

  CHECK(ss.str() == R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: intervals
    intervals: [[-8.5, 8.5], [-3.5, 5.5], null]
  - kind: intervals
    intervals: [[20, 80], [40, 60], null]
sufficient_coalitions:
  - &coalitions
    kind: weights
    criterion_weights: [0.5, 0.5]
  - *coalitions
  - *coalitions
)");

  // CHECK(Model::load(problem, ss) == model);
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
    {SufficientCoalitions(SufficientCoalitions::Roots(problem, {}))},
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
      SufficientCoalitions(SufficientCoalitions::Roots(problem, {{0}, {1, 2}})),
      SufficientCoalitions(SufficientCoalitions::Roots(problem, {{0}, {1, 2}})),
      SufficientCoalitions(SufficientCoalitions::Roots(problem, {{0}, {1, 2}})),
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
      SufficientCoalitions(SufficientCoalitions::Roots(problem, {{}})),
      SufficientCoalitions(SufficientCoalitions::Roots(problem, {{0}, {1}, {2}})),
      SufficientCoalitions(SufficientCoalitions::Roots(problem, {{0}, {1, 2}})),
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
      - []
  - kind: roots
    upset_roots:
      - [0]
      - [1]
      - [2]
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
