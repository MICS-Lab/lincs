// Copyright 2021 Vincent Jacques

#ifndef IMPROVE_WEIGHTS_HPP_
#define IMPROVE_WEIGHTS_HPP_

#include <ortools/lp_data/lp_data.h>

#include <memory>

#include "problem.hpp"


namespace ppl::improve_weights {

typedef unsigned int uint;

std::shared_ptr<operations_research::glop::LinearProgram> make_verbose_linear_program(
  const float epsilon, const Models<Host>&, uint model_index);

void improve_weights(Models<Host>*);

}  // namespace ppl::improve_weights

#endif  // IMPROVE_WEIGHTS_HPP_
