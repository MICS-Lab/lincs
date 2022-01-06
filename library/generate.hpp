// Copyright 2021-2022 Vincent Jacques

#ifndef GENERATE_HPP_
#define GENERATE_HPP_

#include <random>

#include "io.hpp"

namespace ppl::generate {

io::Model model(std::mt19937* gen, uint criteria_count, uint categories_count);

io::LearningSet learning_set(std::mt19937* gen, const io::Model& model, uint alternatives_count);

}  // namespace ppl::generate

#endif  // GENERATE_HPP_
