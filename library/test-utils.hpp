// Copyright 2021-2022 Vincent Jacques

#ifndef TEST_UTILS_HPP_
#define TEST_UTILS_HPP_

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "problem.hpp"

namespace ppl {

Domain<Host> make_domain(
  uint categories_count,
  const std::vector<std::pair<std::vector<float>, uint>>& alternatives_);

Models<Host> make_models(
  const Domain<Host>& domain,
  const std::vector<std::pair<std::vector<std::vector<float>>, std::vector<float>>>& models_);

}  // namespace ppl

#endif  // TEST_UTILS_HPP_
