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

  explicit PythonOutputDevice(bp::object out_file_) : out_file(out_file_) {}

  std::streamsize write(const char* s, std::streamsize n) {
    out_file.attr("write")(std::string(s, n));
    return n;
  }

  private:
  bp::object out_file;
};

void dump_domain(const plad::Domain& domain, bp::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  domain.dump(out_stream);
}

class PythonInputDevice : public boost::iostreams::source {
  public:

  explicit PythonInputDevice(bp::object in_file_) : in_file(in_file_) {}

  std::streamsize read(char* s, std::streamsize n) {
    std::string str = bp::extract<std::string>(in_file.attr("read")(n));
    std::copy(str.begin(), str.end(), s);
    return str.size();
  }

  private:
  bp::object in_file;
};

plad::Domain load_domain(bp::object& in_file) {
  boost::iostreams::stream<PythonInputDevice> in_stream(in_file);
  return plad::Domain::load(in_stream);
}

// https://stackoverflow.com/a/15940413/905845
struct iterable_converter {
  template <typename Container>
  iterable_converter& from_python() {
    bp::converter::registry::push_back(
      &iterable_converter::convertible,
      &iterable_converter::construct<Container>,
      bp::type_id<Container>());

    return *this;
  }

  static void* convertible(PyObject* object) {
    return PyObject_GetIter(object) ? object : NULL;
  }

  template <typename Container>
  static void construct(
    PyObject* object,
    bp::converter::rvalue_from_python_stage1_data* data
  ) {
    bp::handle<> handle(bp::borrowed(object));

    typedef bp::converter::rvalue_from_python_storage<Container> storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    typedef bp::stl_input_iterator<typename Container::value_type> iterator;

    new (storage) Container(iterator(bp::object(handle)), iterator());
    data->convertible = storage;
  }
};

}  // namespace

BOOST_PYTHON_MODULE(libplad) {
  iterable_converter()
    .from_python<std::vector<plad::Domain::Category>>()
    .from_python<std::vector<plad::Domain::Criterion>>()
  ;

  // @todo Use magic_enum to automate the declaration of enum values
  bp::enum_<plad::Domain::Criterion::ValueType>("ValueType")
    .value("real", plad::Domain::Criterion::ValueType::real)
  ;

  bp::enum_<plad::Domain::Criterion::CategoryCorrelation>("CategoryCorrelation")
    .value("growing", plad::Domain::Criterion::CategoryCorrelation::growing)
  ;

  bp::class_<plad::Domain::Criterion>("Criterion", bp::init<std::string, plad::Domain::Criterion::ValueType, plad::Domain::Criterion::CategoryCorrelation>())
    .def_readwrite("name", &plad::Domain::Criterion::name)
    .def_readwrite("value_type", &plad::Domain::Criterion::value_type)
    .def_readwrite("category_correlation", &plad::Domain::Criterion::category_correlation)
  ;

  bp::class_<plad::Domain::Category>("Category", bp::init<std::string>())
    .def_readwrite("name", &plad::Domain::Category::name)
  ;

  bp::class_<plad::Domain>("Domain", bp::init<std::vector<plad::Domain::Criterion>, std::vector<plad::Domain::Category>>())
    .def("dump", &dump_domain)
  ;

  bp::def("load_domain", &load_domain);
}
