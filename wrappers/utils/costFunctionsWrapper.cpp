#include <pybind11/pybind11.h>
#include "utils/costFunctionsWrapper.h"

namespace py = pybind11;

namespace FittingAlgorithms {

void registerCostFunctions(py::module_& module) {
    // Registrar las funciones de coste
    module.def("squaredError", &squaredError, "Squared error cost function");
    module.def("relativeError", &relativeError, "Relative error cost function");
    module.def("squaredLogarithmicError", &squaredLogarimicError, "Squared logarithmic error cost function");
}

}  // namespace FittingAlgorithms
