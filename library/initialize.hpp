// Copyright 2021 Vincent Jacques

#ifndef INITIALIZE_HPP_
#define INITIALIZE_HPP_

#include <vector>

#include "problem.hpp"


namespace ppl {

class ModelsInitializer {
 public:
  ModelsInitializer() {}

 public:
  void initialize(
    Models<Host>* models,
    std::vector<uint>::const_iterator model_indexes_begin,
    std::vector<uint>::const_iterator model_indexes_end);
};

}  // namespace ppl

#endif  // INITIALIZE_HPP_
