#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

namespace FittingAlgorithms{
  namespace GaussNewton {
    void registerGNParameters(py::module_& module);    
    void registerFitResult(py::module_& module);
    void registerFitScalar(py::module_& module);
    void registerFitVector(py::module_& module);
    void registerGaussNewton(py::module_& module);
  }
}
