#pragma once
#include <pybind11/pybind11.h>
#include "utils/costFunctions.h"

namespace py = pybind11;

namespace FittingAlgorithms {
  void registerCostFunctions(py::module_& module);
}
