// Copyright 2021 Vincent Jacques

#ifndef ASSIGN_HPP_
#define ASSIGN_HPP_

#include "problem.hpp"


namespace ppl {

__host__ __device__
uint get_assignment(const ModelsView&, uint model_index, uint alternative_index);
uint get_assignment(const Models<Host>&, uint model_index, uint alternative_index);

// Accuracy is returned as an integer between `0` and `models.domain.alternatives_count`.
// (To get the accuracy described in the thesis, it should be devided by `models.domain.alternatives_count`)
uint get_accuracy(const Models<Host>&, uint model_index);
uint get_accuracy(const Models<Device>&, uint model_index);

}  // namespace ppl

#endif  // ASSIGN_HPP_
