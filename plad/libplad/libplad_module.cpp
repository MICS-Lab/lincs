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

  explicit PythonOutputDevice(boost::python::object out_file_) : out_file(out_file_) {}

  std::streamsize write(const char* s, std::streamsize n) {
    out_file.attr("write")(std::string(s, n));
    return n;
  }

  private:
  boost::python::object out_file;
};

void dump_domain(const plad::Domain& domain, bp::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  domain.dump(out_stream);
}

class PythonInputDevice : public boost::iostreams::source {
  public:

  explicit PythonInputDevice(boost::python::object in_file_) : in_file(in_file_) {}

  std::streamsize read(char* s, std::streamsize n) {
    std::string str = bp::extract<std::string>(in_file.attr("read")(n));
    std::copy(str.begin(), str.end(), s);
    return str.size();
  }

  private:
  boost::python::object in_file;
};

plad::Domain load_domain(bp::object& in_file) {
  boost::iostreams::stream<PythonInputDevice> in_stream(in_file);
  return plad::Domain::load(in_stream);
}

}  // namespace

BOOST_PYTHON_MODULE(libplad) {
  bp::class_<plad::Domain>("Domain", bp::init<int, int>())
    .def("dump", &dump_domain)
  ;

  bp::def("load_domain", &load_domain);
}
