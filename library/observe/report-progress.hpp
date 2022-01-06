// Copyright 2021-2022 Vincent Jacques

#ifndef OBSERVE_REPORT_PROGRESS_HPP_
#define OBSERVE_REPORT_PROGRESS_HPP_


#include <iostream>

#include "../observe.hpp"

namespace ppl {

class ReportProgress : public LearningObserver {
 public:
  void after_main_iteration(int iteration_index, int best_accuracy, const Models<Host>& models) override {
    std::cerr << "After iteration n°" << iteration_index << ": best accuracy = " <<
      best_accuracy << "/" << models.get_view().domain.learning_alternatives_count << std::endl;
  }
};

}  // namespace ppl

#endif  // OBSERVE_REPORT_PROGRESS_HPP_
