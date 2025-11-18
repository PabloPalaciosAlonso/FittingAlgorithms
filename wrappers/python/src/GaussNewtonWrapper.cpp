#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "GaussNewton.h"
#include "registerDefinitions.h"
#include "wrapperUtils.h"
#include "utils/costFunctions.h"

namespace py = pybind11;

namespace FittingAlgorithms {
  namespace GaussNewton {

    void registerParameters(py::module_& module) {
      py::class_<Parameters>(module, "Parameters")
        .def(py::init<>())
        .def_readwrite("maxIterations", &Parameters::maxIterations)
        .def_readwrite("tolerance", &Parameters::tolerance)
        .def_readwrite("regularization", &Parameters::regularization)
        .def_readwrite("printSteps", &Parameters::printSteps);
    }

    template <typename T>
    ModelFunction<T> createModelFunction(py::function model) {
      return [model](const T& x,
                     const std::map<std::string, double>& params,
                     const std::map<std::string, double>& extraParams) -> double {
        py::object result = model(x, params, extraParams);
        return result.cast<double>();
      };
    }

    CostFunction createCostFunction(py::function costFunction) {
      return [costFunction](double ydata, double ypred) -> double {
        return costFunction(ydata, ypred).cast<double>();
      };
    }

    template <typename T>
    void registerTypedFit(py::module_& module, const std::string& description) {
      module.def("fit", [](std::vector<T>& xdata_in,
                           std::vector<double>& ydata_in,
                           py::function model,
                           std::map<std::string, double>& initialGuesses,
                           Parameters& gnParams,
                           py::function costFunction,
                           std::map<std::string, double>& extraParameters) {
        // Crear funciones adaptadas
        auto cpp_model = createModelFunction<T>(model);
        auto cpp_costFunction = createCostFunction(costFunction);

        // Llamar a fit con las funciones adaptadas
        auto result = fit(xdata_in, ydata_in, cpp_model, initialGuesses,
                          gnParams, cpp_costFunction, extraParameters);

        return py::make_tuple(result.parameters, result.errors);
      },
      py::arg("xdata_in"),
      py::arg("ydata_in"),
      py::arg("model"),
      py::arg("initialGuesses"),
      py::arg("gnParams") = Parameters(),
      py::arg("costFunction") = py::cpp_function(squaredError),
      py::arg("extraParameters") = std::map<std::string, double>{},
      description.c_str());
    }

    void registerFittingFunctions(py::module_& module) {
      registerTypedFit<double>(module, "Fit parameters using the Gauss-Newton algorithm for scalar data.");
      registerTypedFit<std::vector<double>>(module, "Fit parameters using the Gauss-Newton algorithm for multidimensional data.");
    }

    void registerGaussNewton(py::module_& module) {
      registerParameters(module);
      registerFittingFunctions(module);
    }
  }
}
