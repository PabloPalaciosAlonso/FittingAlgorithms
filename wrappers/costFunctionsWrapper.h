#pragma once
#include <pybind11/pybind11.h>
#include "utils/costFunctions.h"

namespace py = pybind11;

namespace FittingAlgorithms {

void registerCostFunctions(py::module_& module) {
    // Registrar las funciones de coste
    module.def("squared_error", &squaredError, "Squared error cost function");
    module.def("relative_error", &relativeError, "Relative error cost function");
    module.def("squared_logarithmic_error", &squaredLogarimicError, "Squared logarithmic error cost function");
}

}  // namespace FittingAlgorithms
