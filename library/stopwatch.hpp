// Copyright 2021 Vincent Jacques

#ifndef STOPWATCH_HPP_
#define STOPWATCH_HPP_

#include <string>
#include <chrono>  // NOLINT(build/c++11)


/*
Simplistic stopwatch that prints the time elapsed between its creation
and destruction on the standard error.
*/
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
// Utility macro to create a Stopwatch with a non-colliding variable name.
#define STOPWATCH(name) Stopwatch stopwatch##__line__(name)
#endif

#endif  // STOPWATCH_HPP_
