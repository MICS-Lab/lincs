// Copyright 2021-2022 Vincent Jacques

#ifndef OBSERVE_DUMP_INTERMEDIATE_MODELS_HPP_
#define OBSERVE_DUMP_INTERMEDIATE_MODELS_HPP_

#include <iostream>

#include "../observe.hpp"


namespace ppl {

class DumpIntermediateModels : public LearningObserver {
 public:
  explicit DumpIntermediateModels(std::ostream& stream_);

  void after_main_iteration(int iteration_index, int best_accuracy, const Models<Host>&) override;

 private:
  std::ostream& stream;
};

}  // namespace ppl

#endif  // OBSERVE_DUMP_INTERMEDIATE_MODELS_HPP_
