// Copyright 2023-2024 Vincent Jacques

#include <Python.h>
// https://bugs.python.org/issue36020#msg371558
#undef snprintf
#undef vsnprintf

#include <boost/python.hpp>

#include "../chrones.hpp"

#ifndef DOCTEST_CONFIG_DISABLE
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif
#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


CHRONABLE("lincs");

// @todo(Project management, later) Consider using pybind11, which advertises itself as an evolved and simplified version of Boost.Python
namespace bp = boost::python;

namespace lincs {

void enroll_converters();
void define_io_classes();
void define_generation_functions();
void define_learning_classes();

}  // namespace lincs

BOOST_PYTHON_MODULE(liblincs) {
  bp::docstring_options docstring_options(true, true, false);

  lincs::enroll_converters();
  lincs::define_io_classes();
  lincs::define_generation_functions();
  lincs::define_learning_classes();
}
