// Copyright 2021 Vincent Jacques

#ifndef LEARN_HPP_
#define LEARN_HPP_

#include "io.hpp"
#include "randomness.hpp"


namespace ppl::learn {

io::Model learn_from(const RandomSource& random, const io::LearningSet& learning_set);

}  // namespace ppl::learn

#endif  // LEARN_HPP_
