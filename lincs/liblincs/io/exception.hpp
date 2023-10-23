// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__EXCEPTION_HPP
#define LINCS__IO__EXCEPTION_HPP

#include <stdexcept>


namespace lincs {

struct DataValidationException : public std::runtime_error {
  DataValidationException(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace lincs

#endif  // LINCS__IO__EXCEPTION_HPP
