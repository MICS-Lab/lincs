// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__EXCEPTION_HPP
#define LINCS__LEARNING__EXCEPTION_HPP

#include <stdexcept>


namespace lincs {

struct LearningFailureException : public std::runtime_error {
  LearningFailureException(const char* message) : std::runtime_error(message) {}
};

}  // namespace lincs

#endif  // LINCS__LEARNING__EXCEPTION_HPP
