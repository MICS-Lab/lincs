#include "plad.hpp"

#include <Python.h>


static PyObject* hello_(PyObject* self, PyObject* args) {
    const char* arg;
    if(!PyArg_ParseTuple(args, "s", &arg)) {
        return NULL;
    }

    return PyUnicode_FromString(plad::hello(arg).c_str());
}

static PyMethodDef methods[] = {
    {"hello",  hello_, METH_VARARGS, "Say hello"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "libplad",
    "plad learns and decides",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_libplad(void) {
    return PyModule_Create(&module);
}
