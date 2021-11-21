// Copyright 2021 Vincent Jacques

#ifndef STOPWATCH_HPP_
#define STOPWATCH_HPP_

#include <string>
#include <chrono>  // NOLINT(build/c++11)


struct Stopwatch {
  explicit Stopwatch(const std::string& name);
  ~Stopwatch();

 private:
  const std::string _name;
  const std::chrono::steady_clock::time_point _start_time;
};

#ifdef NOSTOPWATCH
#define STOPWATCH(name)
#else
#define STOPWATCH(name) Stopwatch stopwatch##__line__(name)
#endif

#endif  // STOPWATCH_HPP_
