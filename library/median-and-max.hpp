// Copyright 2021 Vincent Jacques

#ifndef MEDIAN_AND_MAX_HPP_
#define MEDIAN_AND_MAX_HPP_

#include <algorithm>


template<typename RandomIt, class Compare>
void ensure_median_and_max(RandomIt begin, RandomIt end, Compare comp) {
  auto len = end - begin;
  if (len == 0) return;
  // Ensure max
  std::nth_element(begin, begin + len - 1, end, comp);
  // Ensure median, not touching max
  std::nth_element(begin, begin + len / 2, begin + len - 1, comp);
}

#endif  // MEDIAN_AND_MAX_HPP_