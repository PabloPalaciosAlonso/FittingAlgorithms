#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ParallelTempering.h"
#include "registerDefinitions.h"
#include "utils/costFunctions.h"
#include "wrapperUtils.h"

namespace py = pybind11;

namespace FittingAlgorithms {
  namespace ParallelTempering {

    void registerParameters(py::module_& module) {
      py::class_<Parameters>(module, "Parameters")
        .def(py::init<>())
        .def_readwrite("maxIterations", &Parameters::maxIterations)
        .def_readwrite("numStepsSwap", &Parameters::numStepsSwap)
        .def_readwrite("printSteps", &Parameters::printSteps)
        .def_readwrite("numStepsFinish", &Parameters::numStepsFinish)
        .def_readwrite("tolerance", &Parameters::tolerance)
        .def_readwrite("temperatures", &Parameters::temperatures)
        .def_readwrite("jumpSize", &Parameters::jumpSize);
    }

    template <typename T>
    void registerTypedFit(py::module_& module, const std::string& description) {
      module.def("fit", [](std::vector<T>& xdata_in,
                           std::vector<double>& ydata_in,
                           py::function model,
                           std::vector<StringDoubleMap>& initialGuesses,
                           Parameters& ptParams,
                           py::function costFunction,
                           StringDoubleMap extraParameters) {
        
        auto cpp_model        = createModelFunction<T>(model);
        auto cpp_costFunction = createCostFunction(costFunction);

        //GIL do not allow to run python functions in different threads.
        //I don't know how to solve that
        ptParams.numThreads = 1;
        
        auto result = fit(xdata_in, ydata_in, cpp_model, initialGuesses,
                          ptParams, cpp_costFunction, extraParameters);

        //TODO: Think a way to stimate the error
        return py::make_tuple(result, 0);

      },
                 py::arg("xdata_in"),
                 py::arg("ydata_in"),
                 py::arg("model"),
                 py::arg("initialGuesses"),
                 py::arg("ptParams") = Parameters(),
                 py::arg("costFunction") = py::cpp_function(squaredError),
                 py::arg("extraParameters") = StringDoubleMap{},
                 description.c_str());
      
            module.def("fit", [module](std::vector<T>& xdata_in,
                                 std::vector<double>& ydata_in,
                                 py::function model,
                                 StringDoubleMap& initialGuess,
                                 Parameters& ptParams,
                                 py::function costFunction,
                                 StringDoubleMap extraParameters) {
        
        std::vector<StringDoubleMap> guessesVector(ptParams.temperatures.size(), initialGuess);
        
        return module.attr("fit")(xdata_in, ydata_in, model, guessesVector, ptParams, costFunction, extraParameters);
      },
                 py::arg("xdata_in"),
                 py::arg("ydata_in"),
                 py::arg("model"),
                 py::arg("initialGuess"),
                 py::arg("ptParams") = Parameters(),
                 py::arg("costFunction") = py::cpp_function(squaredError),
                 py::arg("extraParameters") = StringDoubleMap{},
                 (description + " (with a single initial guess)").c_str());
    }
    
    
    void registerFittingFunctions(py::module_& module) {
      registerTypedFit<double>(module, "Fit parameters using the Parallel-Tempering algorithm for scalar data.");
      registerTypedFit<std::vector<double>>(module, "Fit parameters using the Parallel-Tempering algorithm for multidimensional data.");
    }

    void registerParallelTempering(py::module_& module) {
      registerParameters(module);
      registerFittingFunctions(module);
    }
  }
}
