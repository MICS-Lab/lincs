// Copyright 2021 Vincent Jacques

#include "stopwatch.hpp"

#include <iostream>


Stopwatch::Stopwatch(const std::string& name) :
  _name(name),
  _start_time(std::chrono::steady_clock::now()) {}

Stopwatch::~Stopwatch() {
  const std::chrono::steady_clock::time_point stop_time = std::chrono::steady_clock::now();
  const uint64_t duration = std::chrono::nanoseconds(stop_time - _start_time).count() / 1000;
  std::cerr << "Stopwatch " << _name << ": " << duration << "µs" << std::endl;
}
