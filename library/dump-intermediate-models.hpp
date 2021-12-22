// Copyright 2021 Vincent Jacques

#ifndef DUMP_INTERMEDIATE_MODELS_HPP_
#define DUMP_INTERMEDIATE_MODELS_HPP_

#include <iostream>

#include "learning.hpp"


namespace ppl {

class IntermediateModelsDumper : public Learning::Observer {
 public:
  explicit IntermediateModelsDumper(std::ostream& stream_);

  void after_main_iteration(int iteration_index, int best_accuracy, const Models<Host>&) override;

 private:
  std::ostream& stream;
};

}  // namespace ppl

#endif  // DUMP_INTERMEDIATE_MODELS_HPP_
