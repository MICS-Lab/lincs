#include "plad.hpp"

#include <iostream>

#include <Python.h>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>
#include <magic_enum.hpp>


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

void dump_model(const plad::Model& model, bp::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  model.dump(out_stream);
}

void dump_alternatives(const plad::AlternativesSet& alternatives, bp::object& out_file) {
  boost::iostreams::stream<PythonOutputDevice> out_stream(out_file);
  alternatives.dump(out_stream);
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

plad::Model load_model(plad::Domain* domain, bp::object& in_file) {
  boost::iostreams::stream<PythonInputDevice> in_stream(in_file);
  return plad::Model::load(domain, in_stream);
}

plad::AlternativesSet load_alternatives(plad::Domain* domain, bp::object& in_file) {
  boost::iostreams::stream<PythonInputDevice> in_stream(in_file);
  return plad::AlternativesSet::load(domain, in_stream);
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

template <typename T>
void auto_enum(const std::string& name) {
  auto e = bp::enum_<T>(name.c_str());
  for(T value : magic_enum::enum_values<T>()) {
    e.value(std::string(magic_enum::enum_name(value)).c_str(), value);
  }
}

BOOST_PYTHON_MODULE(libplad) {
  iterable_converter()
    .from_python<std::vector<float>>()
    .from_python<std::vector<plad::Domain::Category>>()
    .from_python<std::vector<plad::Domain::Criterion>>()
    .from_python<std::vector<plad::Model::Boundary>>()
    .from_python<std::vector<plad::Model::SufficientCoalitions>>()
    .from_python<std::vector<plad::Alternative>>()
  ;

  // @todo Decide wether we nest types or not, use the same nesting in Python and C++
  auto_enum<plad::Domain::Criterion::ValueType>("ValueType");
  auto_enum<plad::Domain::Criterion::CategoryCorrelation>("CategoryCorrelation");
  auto_enum<plad::Model::SufficientCoalitions::Kind>("SufficientCoalitionsKind");

  bp::class_<plad::Domain::Criterion>("Criterion", bp::init<std::string, plad::Domain::Criterion::ValueType, plad::Domain::Criterion::CategoryCorrelation>())
    .def_readwrite("name", &plad::Domain::Criterion::name)
    .def_readwrite("value_type", &plad::Domain::Criterion::value_type)
    .def_readwrite("category_correlation", &plad::Domain::Criterion::category_correlation)
  ;

  bp::class_<plad::Domain::Category>("Category", bp::init<std::string>())
    .def_readwrite("name", &plad::Domain::Category::name)
  ;

  bp::class_<std::vector<plad::Domain::Category>>("categories_vector")
    .def(bp::vector_indexing_suite<std::vector<plad::Domain::Category>>())
  ;
  bp::class_<std::vector<plad::Domain::Criterion>>("criteria_vector")
    .def(bp::vector_indexing_suite<std::vector<plad::Domain::Criterion>>())
  ;
  bp::class_<plad::Domain>("Domain", bp::init<std::vector<plad::Domain::Criterion>, std::vector<plad::Domain::Category>>())
    .def_readwrite("criteria", &plad::Domain::criteria)
    .def_readwrite("categories", &plad::Domain::categories)
    .def("dump", &dump_domain)
  ;
  bp::def("load_domain", &load_domain);

  bp::class_<plad::Model::SufficientCoalitions>("SufficientCoalitions", bp::init<plad::Model::SufficientCoalitions::Kind, std::vector<float>>())
    .def_readwrite("kind", &plad::Model::SufficientCoalitions::kind)
    .def_readwrite("criterion_weights", &plad::Model::SufficientCoalitions::criterion_weights)
  ;

  bp::class_<plad::Model::Boundary>("Boundary", bp::init<std::vector<float>, plad::Model::SufficientCoalitions>())
    .def_readwrite("profile", &plad::Model::Boundary::profile)
    .def_readwrite("sufficient_coalitions", &plad::Model::Boundary::sufficient_coalitions)
  ;

  bp::class_<plad::Model>("Model", bp::init<plad::Domain*, const std::vector<plad::Model::Boundary>&>())
    .def_readwrite("boundaries", &plad::Model::boundaries)
    .def("dump", &dump_model)
  ;
  bp::def("load_model", &load_model);

  bp::class_<plad::Alternative>("Alternative", bp::init<std::string, std::vector<float>, std::string>())
    .def_readwrite("name", &plad::Alternative::name)
    .def_readwrite("profile", &plad::Alternative::profile)
    .def_readwrite("category", &plad::Alternative::category)
  ;

  bp::class_<plad::AlternativesSet>("AlternativesSet", bp::init<plad::Domain*, const std::vector<plad::Alternative>&>())
    .def_readwrite("alternatives", &plad::AlternativesSet::alternatives)
    .def("dump", &dump_alternatives)
  ;
  bp::def("load_alternatives", &load_alternatives);
}
