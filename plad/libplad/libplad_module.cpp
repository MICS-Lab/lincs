#include "plad.hpp"

#include <Python.h>

#include <boost/python.hpp>


namespace bp = boost::python;

BOOST_PYTHON_MODULE(libplad) {
  bp::def("hello", &plad::hello);
}
