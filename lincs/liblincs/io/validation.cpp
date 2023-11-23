// Copyright 2023 Vincent Jacques

#include "validation.hpp"

#include "../chrones.hpp"
#include "../vendored/valijson/adapters/yaml_cpp_adapter.hpp"
#include "../vendored/valijson/schema_parser.hpp"
#include "../vendored/valijson/validator.hpp"


namespace lincs {

JsonValidator::JsonValidator(const YAML::Node& schema_node) {
  valijson::SchemaParser schema_parser;
  valijson::adapters::YamlCppAdapter schema_adapter(schema_node);
  schema_parser.populateSchema(schema_adapter, schema);
}

void JsonValidator::validate(const YAML::Node& document) const {
  CHRONE();

  valijson::adapters::YamlCppAdapter document_adapter(document);

  valijson::Validator validator;
  valijson::ValidationResults results;
  if (!validator.validate(schema, document_adapter, &results)) {
    std::ostringstream oss;
    oss << "JSON validation failed:";

    valijson::ValidationResults::Error error;
    while (results.popError(error)) {
      oss << "\n -";
      for (const std::string &contextElement : error.context) {
          oss << " " << contextElement;
      }
      oss << ": " << error.description;
    }

    throw DataValidationException(oss.str());
  }
}

}  // namespace lincs
