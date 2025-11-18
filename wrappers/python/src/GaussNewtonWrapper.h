#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace FittingAlgorithms{
  namespace GaussNewton {
    void registerGaussNewton(py::module_& module);
  }
}
