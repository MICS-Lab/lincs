// Copyright 2023 Vincent Jacques

#include "validation.hpp"


namespace lincs {

JsonValidator::JsonValidator(const YAML::Node& schema_node) {
  valijson::SchemaParser schema_parser;
  valijson::adapters::YamlCppAdapter schema_adapter(schema_node);
  schema_parser.populateSchema(schema_adapter, schema);
}

void JsonValidator::validate(const YAML::Node& document) const {
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

    throw JsonValidationException(oss.str());
  }
}

}  // namespace lincs
