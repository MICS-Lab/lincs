#include "plad.hpp"

#include <iostream>

#include <Python.h>

#include <boost/python.hpp>
#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>


namespace bp = boost::python;

namespace {
  class PythonOutputDevice : public boost::iostreams::sink {
   public:

    explicit PythonOutputDevice(boost::python::object out_file) : out_file_(out_file) {}

    std::streamsize write(const char* s, std::streamsize n) {
      out_file_.attr("write")(std::string(s, n));
      return n;
    }

   private:
    boost::python::object out_file_;
  };

  void dump_domain(const plad::Domain& domain, bp::object& out_file) {
    boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
    domain.dump(out_stream);
  }
}  // namespace

BOOST_PYTHON_MODULE(libplad) {
  bp::class_<plad::Domain>("Domain")
    .def("dump", &dump_domain)
  ;
}
