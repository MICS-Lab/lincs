// Copyright 2023 Vincent Jacques

#ifndef LINCS__MEDIAN_AND_MAX_HPP
#define LINCS__MEDIAN_AND_MAX_HPP

#include <algorithm>

/*
Ensure that the median and maximum values of the range [begin, end[ are
in the correct positions (middle and last).
Also ensure that all values below the median are before the median,
and all values above the median are after the median.
*/
template<typename RandomIt, class Compare>
void ensure_median_and_max(RandomIt begin, RandomIt end, Compare comp) {
  auto len = end - begin;
  if (len == 0) return;
  // Ensure max
  std::nth_element(begin, begin + len - 1, end, comp);
  // Ensure median, not touching max
  std::nth_element(begin, begin + len / 2, begin + len - 1, comp);
}

#endif  // LINCS__MEDIAN_AND_MAX_HPP
