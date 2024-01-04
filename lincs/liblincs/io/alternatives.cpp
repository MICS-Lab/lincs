// Copyright 2023 Vincent Jacques

#include "alternatives.hpp"

#include "../chrones.hpp"
#include "../vendored/rapidcsv.h"
#include "validation.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace lincs {

// @todo(Project management, v1.1) Consider taking 'alternatives_' by rvalue reference and moving it in 'alternatives'
Alternatives::Alternatives(const Problem& problem, const std::vector<Alternative>& alternatives_) :
  alternatives(alternatives_)
{
  const unsigned criteria_count = problem.get_criteria().size();
  for (const auto& alternative : alternatives) {
    validate(
      alternative.get_profile().size() == problem.get_criteria().size(),
      "The profile of an alternative must have as many performances as there are criteria in the problem");
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      validate(
        alternative.get_profile()[criterion_index].get_value_type() == problem.get_criteria()[criterion_index].get_value_type(),
        "The type of the performance of an alternative must match the type of the criterion in the problem");
    }
  }
}

void Alternatives::dump(const Problem& problem, std::ostream& os) const {
  CHRONE();

  const unsigned criteria_count = problem.get_criteria().size();
  const unsigned alternatives_count = alternatives.size();

  rapidcsv::SeparatorParams separator_params;
  separator_params.mHasCR = false;  // Fix line ends on Windows (without this line, they are "\r\r\n")

  rapidcsv::Document doc(
    std::string(),
    rapidcsv::LabelParams(),
    separator_params,
    rapidcsv::ConverterParams(),
    rapidcsv::LineReaderParams());

  doc.SetColumnName(0, "name");
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    doc.SetColumnName(criterion_index + 1, problem.get_criteria()[criterion_index].get_name());
  }
  doc.SetColumnName(criteria_count + 1, "category");

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    const Alternative& alternative = alternatives[alternative_index];
    doc.SetCell<std::string>(0, alternative_index, alternative.get_name());
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      dispatch(
        alternative.get_profile()[criterion_index].get(),
        [&doc, criterion_index, alternative_index](const Performance::RealPerformance& perf) {
          doc.SetCell<float>(criterion_index + 1, alternative_index, perf.get_value());
        },
        [&doc, criterion_index, alternative_index](const Performance::IntegerPerformance& perf) {
          doc.SetCell<int>(criterion_index + 1, alternative_index, perf.get_value());
        },
        [&doc, criterion_index, alternative_index](const Performance::EnumeratedPerformance& perf) {
          doc.SetCell<std::string>(criterion_index + 1, alternative_index, perf.get_value());
        }
      );
    }
    if (alternative.get_category_index()) {
      doc.SetCell<std::string>(criteria_count + 1, alternative_index, problem.get_ordered_categories()[*alternative.get_category_index()].get_name());
    }
  }

  doc.Save(os);
}

Alternatives Alternatives::load(const Problem& problem, std::istream& is) {
  CHRONE();

  const unsigned criteria_count = problem.get_criteria().size();
  std::map<std::string, unsigned> category_indexes;
  for (const auto& category: problem.get_ordered_categories()) {
    category_indexes[category.get_name()] = category_indexes.size();
  }

  // I don't know why constructing the rapidcsv::Document directly from 'is' sometimes results in an empty document.
  // So, read the whole stream into a string and construct the document from that.
  std::string s(std::istreambuf_iterator<char>(is), {});
  std::istringstream iss(s);
  rapidcsv::Document doc(
    iss,
    rapidcsv::LabelParams(),
    rapidcsv::SeparatorParams(),
    rapidcsv::ConverterParams(),
    rapidcsv::LineReaderParams(true, '#')  // Skip comments
  );

  const ssize_t name_column_index = doc.GetColumnIdx("name");
  if (name_column_index < 0) {
    throw DataValidationException("Missing column: name");
  }

  std::vector<size_t> criterion_column_indexes;
  criterion_column_indexes.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    ssize_t column_index = doc.GetColumnIdx(problem.get_criteria()[criterion_index].get_name());
    if (column_index < 0) {
      throw DataValidationException("Mismatch: criterion from the problem file not found in the alternatives file");
    }
    criterion_column_indexes.emplace_back(column_index);
  }

  const ssize_t category_column_index = doc.GetColumnIdx("category");
  if (category_column_index < 0) {
    throw DataValidationException("Missing column: category");
  }

  std::vector<Alternative> alternatives;
  const unsigned alternatives_count = doc.GetRowCount();
  alternatives.reserve(alternatives_count);
  for (unsigned row_index = 0; row_index != alternatives_count; ++row_index) {
    std::string name = doc.GetCell<std::string>(name_column_index, row_index);
    std::vector<Performance> profile;
    profile.reserve(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      switch (problem.get_criteria()[criterion_index].get_value_type()) {
        case Criterion::ValueType::real:
          profile.push_back(Performance(Performance::RealPerformance(doc.GetCell<float>(criterion_column_indexes[criterion_index], row_index))));
          break;
        case Criterion::ValueType::integer:
          profile.push_back(Performance(Performance::IntegerPerformance(doc.GetCell<int>(criterion_column_indexes[criterion_index], row_index))));
          break;
        case Criterion::ValueType::enumerated:
          profile.push_back(Performance(Performance::EnumeratedPerformance(doc.GetCell<std::string>(criterion_column_indexes[criterion_index], row_index))));
          break;
      }
    }
    std::string category = doc.GetCell<std::string>(category_column_index, row_index);
    std::optional<unsigned> category_index;
    if (category != "") {
      auto it = category_indexes.find(category);
      if (it == category_indexes.end()) {
        throw DataValidationException("Mismatch: category in the alternatives file not found in the problem file");
      }
      category_index = it->second;
    }
    alternatives.emplace_back(name, profile, category_index);
  }

  return Alternatives{problem, alternatives};
}

TEST_CASE("Dump then load preserves data - real criterion") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
    {{"Category 1"}, {"Category 2"}},
  };

  Alternatives alternatives(
    problem,
    {
      {"Alt 1", {Performance(Performance::RealPerformance(0.5))}, 0},
      {"Alt 2", {Performance(Performance::RealPerformance(0.75))}, 1},
    }
  );

  std::stringstream ss;
  alternatives.dump(problem, ss);

  CHECK(ss.str() == R"(name,"Criterion 1",category
"Alt 1",0.5,"Category 1"
"Alt 2",0.75,"Category 2"
)");

  CHECK(Alternatives::load(problem, ss) == alternatives);
}

TEST_CASE("Dump then load preserves data - numerical values requiring more decimal digits") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
    {{"Category 1"}, {"Category 2"}},
  };

  Alternatives alternatives(problem, {
    {"Alt 1", {Performance(Performance::RealPerformance(0x1.259b36p-6))}, 0},
    {"Alt 2", {Performance(Performance::RealPerformance(0x1.652bf4p-2))}, 1},
    {"Alt 3", {Performance(Performance::RealPerformance(0x1.87662ap-3))}, 1},
  });

  std::stringstream ss;
  alternatives.dump(problem, ss);

  CHECK(ss.str() == R"(name,"Criterion 1",category
"Alt 1",0.017920306,"Category 1"
"Alt 2",0.34880048,"Category 2"
"Alt 3",0.191112831,"Category 2"
)");

  CHECK(Alternatives::load(problem, ss) == alternatives);
}

TEST_CASE("Dump then load preserves data - integer criterion") {
  Problem problem(
    {Criterion("Criterion 1", Criterion::IntegerValues(Criterion::PreferenceDirection::increasing, 0, 10))},
    {{"Category 1"}, {"Category 2"}}
  );

  Alternatives alternatives(
    problem,
    {
      {"Alt 1", {Performance(Performance::IntegerPerformance(5))}, 0},
      {"Alt 2", {Performance(Performance::IntegerPerformance(6))}, 1},
    }
  );

  std::stringstream ss;
  alternatives.dump(problem, ss);

  CHECK(ss.str() == R"(name,"Criterion 1",category
"Alt 1",5,"Category 1"
"Alt 2",6,"Category 2"
)");

  CHECK(Alternatives::load(problem, ss) == alternatives);
}

TEST_CASE("Dump then load preserves data - enumerated criterion") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::EnumeratedValues({"a", "b b", "c", "d"}))},
    {{"Category 1"}, {"Category 2"}},
  };

  Alternatives alternatives(
    problem,
    {
      {"Alt 1", {Performance(Performance::EnumeratedPerformance("a"))}, 0},
      {"Alt 2", {Performance(Performance::EnumeratedPerformance("b b"))}, 1},
    }
  );

  std::stringstream ss;
  alternatives.dump(problem, ss);

  CHECK(ss.str() == R"(name,"Criterion 1",category
"Alt 1",a,"Category 1"
"Alt 2","b b","Category 2"
)");

  CHECK(Alternatives::load(problem, ss) == alternatives);
}

TEST_CASE("Validation error - name column") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss(R"(namee,"Criterion 1",category
"Alt 1",0.5,
)");

  CHECK_THROWS_WITH_AS(
    Alternatives::load(problem, iss),
    "Missing column: name",
    DataValidationException);
}

TEST_CASE("Validation error - category column") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss(R"(name,"Criterion 1",categoryy
"Alt 1",0.5,
)");

  CHECK_THROWS_WITH_AS(
    Alternatives::load(problem, iss),
    "Missing column: category",
    DataValidationException);
}

TEST_CASE("Validation error - criterion name") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss(R"(name,"Criterion A",category
"Alt 1",0.5,
)");

  CHECK_THROWS_WITH_AS(
    Alternatives::load(problem, iss),
    "Mismatch: criterion from the problem file not found in the alternatives file",
    DataValidationException);
}

TEST_CASE("Validation error - category name") {
  Problem problem{
    {Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1))},
    {{"Category 1"}, {"Category 2"}},
  };

  std::istringstream iss(R"(name,"Criterion 1",category
"Alt 1",0.5,Category 3
)");

  CHECK_THROWS_WITH_AS(
    Alternatives::load(problem, iss),
    "Mismatch: category in the alternatives file not found in the problem file",
    DataValidationException);
}

}  // namespace lincs
