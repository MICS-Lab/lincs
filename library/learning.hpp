// Copyright 2021 Vincent Jacques

#ifndef LEARNING_HPP_
#define LEARNING_HPP_

#include <utility>

#include "io.hpp"
#include "randomness.hpp"


namespace ppl::learning {

std::pair<io::Model, uint> learn_from(const RandomSource& random, const io::LearningSet& learning_set);

}  // namespace ppl::learning

#endif  // LEARNING_HPP_
