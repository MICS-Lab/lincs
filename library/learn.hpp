// Copyright 2021 Vincent Jacques

#ifndef LEARN_HPP_
#define LEARN_HPP_

#include <utility>

#include "io.hpp"
#include "randomness.hpp"


namespace ppl::learn {

std::pair<io::Model, uint> learn_from(const RandomSource& random, const io::LearningSet& learning_set);

}  // namespace ppl::learn

#endif  // LEARN_HPP_
