// Copyright 2023-2024 Vincent Jacques

#include <optional>
#include <random>
#include <string>
#include <vector>

#include "../lincs.hpp"
#include "../vendored/lov-e.hpp"
#include "../vendored/pybind11/pybind11.h"
#include "../vendored/pybind11/stl.h"


namespace py = pybind11;

namespace lincs {

void enroll_standard_converters(py::module& m) {
  py::class_<std::mt19937>(
    m,
    "UniformRandomBitsGenerator",
    "Random number generator."
  )
    .def(
      "__call__",
      &std::mt19937::operator(),
      "Generate the next pseudo-random integer."
    )
  ;
}

}  // namespace lincs

namespace lincs {

void enroll_love_converters(py::module& m) {
  py::class_<Array1D<Host, bool>>(m, "Array1D<Host, bool>")
    .def("__len__", &Array1D<Host, bool>::s0)
    .def("__getitem__", [](const Array1D<Host, bool>& c, unsigned i) {
      if (i >= c.s0()) {
        throw pybind11::index_error();
      }
      return c[i];
    })
    .def("__setitem__", [](Array1D<Host, bool>& c, unsigned i, bool v) {
      if (i >= c.s0()) {
        throw pybind11::index_error();
      }
      c[i] = v;
    })
  ;
  py::class_<Array1D<Host, unsigned>>(m, "Array1D<Host, unsigned>")
    .def("__len__", &Array1D<Host, unsigned>::s0)
    .def("__getitem__", [](const Array1D<Host, unsigned>& c, unsigned i) {
      if (i >= c.s0()) {
        throw pybind11::index_error();
      }
      return c[i];
    })
    .def("__setitem__", [](Array1D<Host, unsigned>& c, unsigned i, unsigned v) {
      if (i >= c.s0()) {
        throw pybind11::index_error();
      }
      c[i] = v;
    })
  ;
  py::class_<ArrayView1D<Host, unsigned>>(m, "ArrayView1D<Host, unsigned>")
    .def("__len__", &ArrayView1D<Host, unsigned>::s0)
    .def("__getitem__", [](const ArrayView1D<Host, unsigned>& c, unsigned i) {
      if (i >= c.s0()) {
        throw pybind11::index_error();
      }
      return c[i];
    })
    .def("__setitem__", [](ArrayView1D<Host, unsigned>& c, unsigned i, unsigned v) {
      if (i >= c.s0()) {
        throw pybind11::index_error();
      }
      c[i] = v;
    })
  ;
  py::class_<Array2D<Host, unsigned>>(m, "Array2D<Host, unsigned>")
    .def("__len__", &Array2D<Host, unsigned>::s1)
    .def("__getitem__", [](const Array2D<Host, unsigned>& c, unsigned i) {
      if (i >= c.s1()) {
        throw pybind11::index_error();
      }
      return c[i];
    })
  ;
  py::class_<ArrayView2D<Host, unsigned>>(m, "ArrayView2D<Host, unsigned>")
    .def("__len__", &ArrayView2D<Host, unsigned>::s1)
    .def("__getitem__", [](const ArrayView2D<Host, unsigned>& c, unsigned i) {
      if (i >= c.s1()) {
        throw pybind11::index_error();
      }
      return c[i];
    })
  ;
  py::class_<Array3D<Host, unsigned>>(m, "Array3D<Host, unsigned>")
    .def("__len__", &Array3D<Host, unsigned>::s2)
    .def("__getitem__", [](const Array3D<Host, unsigned>& c, unsigned i) {
      if (i >= c.s2()) {
        throw pybind11::index_error();
      }
      return c[i];
    })
  ;
  py::class_<ArrayView1D<Host, float>>(m, "ArrayView1D<Host, float>")
    .def("__len__", &ArrayView1D<Host, float>::s0)
    .def("__getitem__", [](const ArrayView1D<Host, float>& c, unsigned i) {
      if (i >= c.s0()) {
        throw pybind11::index_error();
      }
      return c[i];
    })
    .def("__setitem__", [](ArrayView1D<Host, float>& c, unsigned i, float v) {
      if (i >= c.s0()) {
        throw pybind11::index_error();
      }
      c[i] = v;
    })
  ;
  py::class_<Array2D<Host, float>>(m, "Array2D<Host, float>")
    .def("__len__", &Array2D<Host, float>::s1)
    .def("__getitem__", [](const Array2D<Host, float>& c, unsigned i) {
      if (i >= c.s1()) {
        throw pybind11::index_error();
      }
      return c[i];
    })
  ;
}

void enroll_converters(py::module& m) {
  enroll_love_converters(m);
  enroll_standard_converters(m);
}

}  // namespace lincs
