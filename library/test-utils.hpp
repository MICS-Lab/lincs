// Copyright 2021 Vincent Jacques

#ifndef TEST_UTILS_HPP_
#define TEST_UTILS_HPP_

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "problem.hpp"


ppl::Domain<Host> make_domain(
  uint categories_count,
  const std::vector<std::pair<std::vector<float>, uint>>& alternatives_);

ppl::Models<Host> make_models(
  const ppl::Domain<Host>& domain,
  const std::vector<std::pair<std::vector<std::vector<float>>, std::vector<float>>>& models_);

#endif  // TEST_UTILS_HPP_
