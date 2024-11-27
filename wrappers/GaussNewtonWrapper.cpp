#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "GaussNewton.h"
#include "GaussNewtonWrapper.h"

namespace py = pybind11;

namespace FittingAlgorithms {
  namespace GaussNewton {

    
    void registerGaussNewton(py::module_& gaussNewtonModule) {
      py::class_<GNParameters>(gaussNewtonModule, "GNParameters")
        .def(py::init<>())
        .def_readwrite("maxIterations", &GNParameters::maxIterations)
        .def_readwrite("tolerance", &GNParameters::tolerance)
        .def_readwrite("regularization", &GNParameters::regularization)
        .def_readwrite("printSteps", &GNParameters::printSteps);
    
    
      // Registrar FitResult
      py::class_<FitResult>(gaussNewtonModule, "FitResult")
        .def_readonly("parameters", &FitResult::parameters)
        .def_readonly("errors", &FitResult::errors);
      
      // Registrar `fitScalar`
      gaussNewtonModule.def("fit", [](std::vector<double>& xdata_in,
                                      std::vector<double>& ydata_in,
                                      GNParameters& gnParams,
                                      py::function model,
                                      std::map<std::string, double>& initialGuesses,
                                      std::map<std::string, double>& extraParameters) {
        // Adaptar la función Python directamente a std::function
        auto cpp_model = [model](const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 const std::map<std::string, double>& params,
                                 const std::map<std::string, double>& extraParams) -> std::vector<double> {
          return model(x, y, params, extraParams).cast<std::vector<double>>();
        };
        
        // Llamar a la función fitScalar
        return fitScalar(xdata_in, ydata_in, gnParams, cpp_model, initialGuesses, extraParameters);
      },
                            py::arg("xdata_in"),
                            py::arg("ydata_in"),
                            py::arg("gnParams"),
                            py::arg("model"),
                            py::arg("initialGuesses"),
                            py::arg("extraParameters"),
                            "Fit parameters using the Gauss-Newton algorithm for scalar data.");
      
      // Registrar `fitVector`
      gaussNewtonModule.def("fit", [](std::vector<std::vector<double>>& xdata_in,
                                      std::vector<double>& ydata_in,
                                      GNParameters& gnParams,
                                      py::function model,
                                      std::map<std::string, double>& initialGuesses,
                                      std::map<std::string, double>& extraParameters) {
        // Adaptar la función Python directamente a std::function
        auto cpp_model = [model](const std::vector<std::vector<double>>& x,
                                 const std::vector<double>& y,
                                 const std::map<std::string, double>& params,
                                 const std::map<std::string, double>& extraParams) -> std::vector<double> {
          return model(x, y, params, extraParams).cast<std::vector<double>>();
        };
        
        // Llamar a la función fitVector
        return fitVector(xdata_in, ydata_in, gnParams, cpp_model, initialGuesses, extraParameters);
      },
                            py::arg("xdata_in"),
                            py::arg("ydata_in"),
                            py::arg("gnParams"),
                            py::arg("model"),
                            py::arg("initialGuesses"),
                            py::arg("extraParameters"),
                            "Fit parameters using the Gauss-Newton algorithm for multidimensional data.");
    }
  }
}
