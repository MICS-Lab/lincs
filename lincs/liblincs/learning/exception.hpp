// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__EXCEPTION_HPP
#define LINCS__LEARNING__EXCEPTION_HPP

#include <exception>


namespace lincs {

class LearningFailureException : public std::exception {
 public:
  const char* what() const noexcept override {
    return "Unable to learn from this dataset. Try to learn with a lower accuracy target? (Not supported by all learning strategies)";
  }
};

}  // namespace lincs

#endif  // LINCS__LEARNING__EXCEPTION_HPP
