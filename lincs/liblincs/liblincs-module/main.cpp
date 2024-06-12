// Copyright 2023-2024 Vincent Jacques

#include "../chrones.hpp"
#include "../vendored/pybind11/pybind11.h"
#include "../vendored/pybind11/stl.h"

#ifndef DOCTEST_CONFIG_DISABLE
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif
#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


CHRONABLE("lincs");

namespace py = pybind11;

namespace lincs {

void enroll_converters(py::module&);
void define_io_classes(py::module&);
void define_generation_functions(py::module&);
void define_learning_classes(py::module&);

}  // namespace lincs

PYBIND11_MODULE(liblincs, m) {
  py::options options;
  options.disable_enum_members_docstring();

  lincs::enroll_converters(m);
  lincs::define_io_classes(m);
  lincs::define_generation_functions(m);
  lincs::define_learning_classes(m);
}
