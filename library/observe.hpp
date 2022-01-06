// Copyright 2021 Vincent Jacques

#ifndef OBSERVE_HPP_
#define OBSERVE_HPP_

#include "problem.hpp"


namespace ppl {

class LearningObserver {
 public:
  virtual ~LearningObserver() {}

  virtual void after_main_iteration(int iteration_index, int best_accuracy, const Models<Host>& models) = 0;
};

}  // namespace ppl

#endif  // OBSERVE_HPP_
