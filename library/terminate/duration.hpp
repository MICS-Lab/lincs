// Copyright 2021 Vincent Jacques

#ifndef TERMINATE_DURATION_HPP_
#define TERMINATE_DURATION_HPP_

#include <chrono>  // NOLINT(build/c++11)

#include "../terminate.hpp"


namespace ppl {

class TerminateAfterDuration : public TerminationStrategy {
  typedef std::chrono::steady_clock clock;

 public:
  explicit TerminateAfterDuration(typename clock::duration max_duration) :
    _max_time(clock::now() + max_duration) {}

  bool terminate(uint /*iteration_index*/, uint /*best_accuracy*/) override {
    return clock::now() >= _max_time;
  }

 private:
  typename clock::time_point _max_time;
};

}  // namespace ppl

#endif  // TERMINATE_DURATION_HPP_
