#include <pybind11/pybind11.h>
#include "utils/costFunctionsWrapper.h"

namespace py = pybind11;

namespace FittingAlgorithms {
  
  void registerCostFunctions(py::module_& module) {
    module.def("squaredError", &squaredError, "Squared error cost function");
    module.def("squaredRelativeError", &squaredRelativeError, "Squared relative error cost function");
    module.def("squaredLogarithmicError", &squaredLogarithmicError, "Squared logarithmic error cost function");
  }
}
