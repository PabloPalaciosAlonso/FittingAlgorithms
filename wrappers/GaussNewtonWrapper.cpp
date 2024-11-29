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
        
    void registerFitScalar(py::module_& module) {
      module.def("fit", [](std::vector<double>& xdata_in,
                           std::vector<double>& ydata_in,
                           GNParameters& gnParams,
                           py::function model,
                           py::function costFunction,
                           std::map<std::string, double>& initialGuesses,
                           std::map<std::string, double>& extraParameters) {
        // Lambda para adaptar el modelo
        auto cpp_model = [model](double x,
                                 const std::map<std::string, double>& params,
                                 const std::map<std::string, double>& extraParams) -> double {
          return model(x, params, extraParams).cast<double>();
        };
        
        // Lambda para adaptar la función de coste
        auto cpp_costFunction = [costFunction](double ydata, double ypred) -> double {
          return costFunction(ydata, ypred).cast<double>();
        };
        
        // Llamar a fitScalar con las lambdas adaptadas
        auto result = fitScalar(xdata_in, ydata_in, gnParams,
                                cpp_model, cpp_costFunction, initialGuesses, extraParameters);

        return py::make_tuple(result.parameters, result.errors);
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
    
    void registerFitScalarNoCostFunction(py::module_& module) {
      module.def("fit", [](std::vector<double>& xdata_in,
                           std::vector<double>& ydata_in,
                           GNParameters& gnParams,
                           py::function model,
                           std::map<std::string, double>& initialGuesses,
                           std::map<std::string, double>& extraParameters) {
        // Crear un lambda para adaptar el modelo de Python al tipo esperado en C++
        auto cpp_model = [model](double x,
                                 const std::map<std::string, double>& params,
                                 const std::map<std::string, double>& extraParams) -> double {
          return model(x, params, extraParams).cast<double>();
        };

        auto result = fitScalar(xdata_in, ydata_in, gnParams,
                                cpp_model, initialGuesses, extraParameters);
        // Llamar a fitScalar con el modelo adaptado
        return py::make_tuple(result.parameters, result.errors);
      },
                 py::arg("xdata_in"),
                 py::arg("ydata_in"),
                 py::arg("gnParams"),
                 py::arg("model"),
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

    void registerFitVectorNoCostFunction(py::module_& module) {
      module.def("fit", [](std::vector<std::vector<double>>& xdata_in,
                           std::vector<double>& ydata_in,
                           GNParameters& gnParams,
                           py::function model,
                           std::map<std::string, double>& initialGuesses,
                           std::map<std::string, double>& extraParameters) {
        return fitVector(xdata_in, ydata_in, gnParams,
                         model.cast<ModelFunction<std::vector<double>>>(),
                         initialGuesses, extraParameters);
    },
                 py::arg("xdata_in"),
                 py::arg("ydata_in"),
                 py::arg("gnParams"),
                 py::arg("model"),
                 py::arg("initialGuesses"),
                 py::arg("extraParameters"),
                 "Fit parameters using the Gauss-Newton algorithm for multidimensional data.");
    }
    
    void registerGaussNewton(py::module_& gaussNewtonModule) {
      registerGNParameters(gaussNewtonModule);
      registerFitScalar(gaussNewtonModule);
      registerFitVector(gaussNewtonModule);
      registerFitScalarNoCostFunction(gaussNewtonModule);
      registerFitVectorNoCostFunction(gaussNewtonModule);
    }
  }
}
