// Copyright 2023 Vincent Jacques

#include "alternatives.hpp"

#include "../chrones.hpp"
#include "../vendored/rapidcsv.h"
#include "exception.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace lincs {

void Alternatives::dump(const Problem& problem, std::ostream& os) const {
  CHRONE();

  const unsigned criteria_count = problem.criteria.size();
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
    doc.SetColumnName(criterion_index + 1, problem.criteria[criterion_index].name);
  }
  doc.SetColumnName(criteria_count + 1, "category");

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    const Alternative& alternative = alternatives[alternative_index];
    doc.SetCell<std::string>(0, alternative_index, alternative.name);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      doc.SetCell<float>(criterion_index + 1, alternative_index, alternative.profile[criterion_index]);
    }
    if (alternative.category_index) {
      doc.SetCell<std::string>(criteria_count + 1, alternative_index, problem.categories[*alternative.category_index].name);
    }
  }

  doc.Save(os);
}

Alternatives Alternatives::load(const Problem& problem, std::istream& is) {
  CHRONE();

  const unsigned criteria_count = problem.criteria.size();
  std::map<std::string, unsigned> category_indexes;
  for (const auto& category: problem.categories) {
    category_indexes[category.name] = category_indexes.size();
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
    ssize_t column_index = doc.GetColumnIdx(problem.criteria[criterion_index].name);
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
    Alternative alternative;
    alternative.name = doc.GetCell<std::string>(name_column_index, row_index);
    alternative.profile.reserve(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      alternative.profile.push_back(doc.GetCell<float>(criterion_column_indexes[criterion_index], row_index));
    }
    std::string category = doc.GetCell<std::string>(category_column_index, row_index);
    if (category != "") {
      auto it = category_indexes.find(category);
      if (it == category_indexes.end()) {
        throw DataValidationException("Mismatch: category in the alternatives file not found in the problem file");
      }
      alternative.category_index = it->second;
    }
    alternatives.push_back(alternative);
  }

  return Alternatives{problem, alternatives};
}

TEST_CASE("Validation error - name column") {
  Problem problem{
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
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
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
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
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
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
    {{"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1}},
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
