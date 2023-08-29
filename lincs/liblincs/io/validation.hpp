// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__VALIDATION_HPP
#define LINCS__IO__VALIDATION_HPP

#include "../vendored/valijson/schema.hpp"
#include "../vendored/yaml-cpp/yaml.h"

namespace lincs {

class JsonValidationException : public std::runtime_error {
 public:
  JsonValidationException(const std::string& message) : std::runtime_error(message) {}
};

class JsonValidator {
 public:
  JsonValidator(const YAML::Node& schema);

  void validate(const YAML::Node& document) const;

 private:
  valijson::Schema schema;
};

}  // namespace lincs

#endif  // LINCS__IO__VALIDATION_HPP
