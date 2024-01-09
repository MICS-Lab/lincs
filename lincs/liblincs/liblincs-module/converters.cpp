// Copyright 2023 Vincent Jacques

#include <Python.h>
// https://bugs.python.org/issue36020#msg371558
#undef snprintf
#undef vsnprintf

#include <optional>
#include <string>
#include <vector>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "../lincs.hpp"


namespace bp = boost::python;

namespace {

template<typename T>
struct std_vector_converter {
  static void* convertible(PyObject* obj) {
    if (PyObject_GetIter(obj)) {
      return obj;
    } else {
      return nullptr;
    }
  }

  static void construct(PyObject* obj, bp::converter::rvalue_from_python_stage1_data* data) {
    bp::handle<> handle(bp::borrowed(obj));

    typedef bp::converter::rvalue_from_python_storage<std::vector<T>> storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    typedef bp::stl_input_iterator<typename std::vector<T>::value_type> iterator;

    new (storage) std::vector<T>(iterator(bp::object(handle)), iterator());
    data->convertible = storage;
  }

  static void enroll() {
    // No need for 'bp::to_python_converter': already implemented by Boost.Python
    bp::converter::registry::push_back(
      &std_vector_converter<T>::convertible,
      &std_vector_converter<T>::construct,
      bp::type_id<std::vector<T>>()
    );
  }
};

template<typename T>
struct std_vector_converter<std::vector<T>> {
  static PyObject* convert(const std::vector<std::vector<T>>& vvv) {
    bp::list result;
    for (const std::vector<T>& vv : vvv) {
      bp::list sublist;
      for (const T& v : vv) {
        sublist.append(v);
      }
      result.append(sublist);
    }
    return bp::incref(result.ptr());
  }

  static void* convertible(PyObject* obj) {
    if (PyObject_GetIter(obj)) {
      return obj;
    } else {
      return nullptr;
    }
  }

  static void construct(PyObject* obj, bp::converter::rvalue_from_python_stage1_data* data) {
    bp::handle<> handle(bp::borrowed(obj));

    typedef bp::converter::rvalue_from_python_storage<std::vector<std::vector<T>>> storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    typedef bp::stl_input_iterator<typename std::vector<std::vector<T>>::value_type> iterator;

    new (storage) std::vector<std::vector<T>>(iterator(bp::object(handle)), iterator());
    data->convertible = storage;
  }

  static void enroll() {
    bp::to_python_converter<std::vector<std::vector<T>>, std_vector_converter<std::vector<T>>>();
    bp::converter::registry::push_back(
      &std_vector_converter<std::vector<T>>::convertible,
      &std_vector_converter<std::vector<T>>::construct,
      bp::type_id<std::vector<std::vector<T>>>()
    );
  }
};

template<typename T>
struct std_optional_converter {
  static PyObject* convert(const std::optional<T>& value) {
    if (value) {
      return bp::incref(bp::object(*value).ptr());
    } else {
      return bp::incref(bp::object().ptr());
    }
  }

  static void* convertible(PyObject* obj) {
    if (obj == Py_None) {
      return new std::optional<T>();
    } else if (PyNumber_Check(obj) || PyUnicode_Check(obj)) {
      return new std::optional<T>(bp::extract<T>(obj));
    } else {
      return nullptr;
    }
  }

  static void construct(PyObject* obj, bp::converter::rvalue_from_python_stage1_data* data) {
    void* storage = reinterpret_cast<bp::converter::rvalue_from_python_storage<std::optional<T>>*>(data)->storage.bytes;
    new (storage) std::optional<T>(*reinterpret_cast<std::optional<T>*>(convertible(obj)));
    data->convertible = storage;
  }

  static void enroll() {
    bp::to_python_converter<std::optional<T>, std_optional_converter<T>>();
    bp::converter::registry::push_back(
      &std_optional_converter<T>::convertible,
      &std_optional_converter<T>::construct,
      bp::type_id<std::optional<T>>()
    );
  }
};

}  // namespace

namespace lincs {

void enroll_standard_converters() {
  std_vector_converter<float>::enroll();
  bp::class_<std::vector<float>>("Iterable[float]").def(bp::vector_indexing_suite<std::vector<float>>());

  std_vector_converter<int>::enroll();
  bp::class_<std::vector<int>>("Iterable[int]").def(bp::vector_indexing_suite<std::vector<int>>());

  std_vector_converter<unsigned>::enroll();

  std_vector_converter<std::string>::enroll();
  bp::class_<std::vector<std::string>>("Iterable[str]").def(bp::vector_indexing_suite<std::vector<std::string>>());

  std_vector_converter<std::vector<unsigned>>::enroll();

  std_vector_converter<lincs::Category>::enroll();
  bp::class_<std::vector<lincs::Category>>("Iterable[Category]").def(bp::vector_indexing_suite<std::vector<lincs::Category>>());

  std_vector_converter<lincs::Criterion>::enroll();
  bp::class_<std::vector<lincs::Criterion>>("Iterable[Criterion]").def(bp::vector_indexing_suite<std::vector<lincs::Criterion>>());

  std_vector_converter<lincs::Criterion::PreferenceDirection>::enroll();
  bp::class_<std::vector<lincs::Criterion::PreferenceDirection>>("Iterable[PreferenceDirection]").def(bp::vector_indexing_suite<std::vector<lincs::Criterion::PreferenceDirection>>());

  std_vector_converter<lincs::Criterion::ValueType>::enroll();
  bp::class_<std::vector<lincs::Criterion::ValueType>>("Iterable[ValueType]").def(bp::vector_indexing_suite<std::vector<lincs::Criterion::ValueType>>());

  std_vector_converter<lincs::AcceptedValues>::enroll();
  bp::class_<std::vector<lincs::AcceptedValues>>("Iterable[AcceptedValues]").def(bp::vector_indexing_suite<std::vector<lincs::AcceptedValues>>());

  std_vector_converter<lincs::SufficientCoalitions>::enroll();
  bp::class_<std::vector<lincs::SufficientCoalitions>>("Iterable[SufficientCoalitions]").def(bp::vector_indexing_suite<std::vector<lincs::SufficientCoalitions>>());

  std_vector_converter<lincs::Performance>::enroll();
  bp::class_<std::vector<lincs::Performance>>("Iterable[Performance]").def(bp::vector_indexing_suite<std::vector<lincs::Performance>>());

  std_vector_converter<lincs::Alternative>::enroll();
  bp::class_<std::vector<lincs::Alternative>>("Iterable[Alternative]").def(bp::vector_indexing_suite<std::vector<lincs::Alternative>>());

  std_vector_converter<lincs::LearnMrsortByWeightsProfilesBreed::TerminationStrategy*>::enroll();

  std_vector_converter<lincs::LearnMrsortByWeightsProfilesBreed::Observer*>::enroll();

  std_optional_converter<float>::enroll();
  std_optional_converter<unsigned>::enroll();
}

}  // namespace lincs
