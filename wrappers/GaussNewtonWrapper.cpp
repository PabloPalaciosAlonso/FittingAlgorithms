#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "GaussNewton.h"
#include "GaussNewtonWrapper.h"

namespace py = pybind11;

namespace FittingAlgorithms {
  namespace GaussNewton {
    
    void registerGNParameters(py::module_& module) {
      py::class_<GNParameters>(module, "GNParameters")
        .def(py::init<>())
        .def_readwrite("maxIterations", &GNParameters::maxIterations)
        .def_readwrite("tolerance", &GNParameters::tolerance)
        .def_readwrite("regularization", &GNParameters::regularization)
        .def_readwrite("printSteps", &GNParameters::printSteps);
    }
    
    void registerFitResult(py::module_& module) {
      py::class_<FitResult>(module, "FitResult")
        .def_readonly("parameters", &FitResult::parameters)
        .def_readonly("errors", &FitResult::errors);
    }
    
    void registerFitScalar(py::module_& module) {
      module.def("fit", [](std::vector<double>& xdata_in,
                           std::vector<double>& ydata_in,
                           GNParameters& gnParams,
                           py::function model,
                           py::function costFunction,
                           std::map<std::string, double>& initialGuesses,
                           std::map<std::string, double>& extraParameters) {
        return fitScalar(xdata_in, ydata_in, gnParams,
                         model.cast<ModelFunction<double>>(),
                         costFunction.cast<CostFunction>(),
                         initialGuesses, extraParameters);
      },
                 py::arg("xdata_in"),
                 py::arg("ydata_in"),
                 py::arg("gnParams"),
                 py::arg("model"),
                 py::arg("costFunction"),
                 py::arg("initialGuesses"),
                 py::arg("extraParameters"),
                 "Fit parameters using the Gauss-Newton algorithm for scalar data.");
    }
    
    void registerFitVector(py::module_& module) {
      module.def("fit", [](std::vector<std::vector<double>>& xdata_in,
                           std::vector<double>& ydata_in,
                           GNParameters& gnParams,
                           py::function model,
                           py::function costFunction,
                           std::map<std::string, double>& initialGuesses,
                           std::map<std::string, double>& extraParameters) {
        return fitVector(xdata_in, ydata_in, gnParams,
                         model.cast<ModelFunction<std::vector<double>>>(),
                         costFunction.cast<CostFunction>(),
                         initialGuesses, extraParameters);
    },
                 py::arg("xdata_in"),
                 py::arg("ydata_in"),
                 py::arg("gnParams"),
                 py::arg("model"),
                 py::arg("costFunction"),
                 py::arg("initialGuesses"),
                 py::arg("extraParameters"),
                 "Fit parameters using the Gauss-Newton algorithm for multidimensional data.");
    }
    
    void registerGaussNewton(py::module_& gaussNewtonModule) {
      registerGNParameters(gaussNewtonModule);
      registerFitResult(gaussNewtonModule);
      registerFitScalar(gaussNewtonModule);
      registerFitVector(gaussNewtonModule);
    }
  }
}
