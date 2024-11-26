#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "../src/GaussNewton.h"

namespace py = pybind11;

// Envolver la función fitParams para exponerla a Python
PYBIND11_MODULE(gauss_newton, m) {
    m.doc() = "Python bindings for the Gauss-Newton fitting algorithm";

    py::class_<Gauss_Newton::GNParameters>(m, "GNParameters")
        .def(py::init<>())
        .def_readwrite("maxIterations", &Gauss_Newton::GNParameters::maxIterations)
        .def_readwrite("tolerance", &Gauss_Newton::GNParameters::tolerance)
        .def_readwrite("regularization", &Gauss_Newton::GNParameters::regularization)
        .def_readwrite("printSteps", &Gauss_Newton::GNParameters::printSteps);

    py::class_<Gauss_Newton::FitResult>(m, "FitResult")
        .def_readonly("parameters", &Gauss_Newton::FitResult::parameters)
        .def_readonly("errors", &Gauss_Newton::FitResult::errors);

    m.def("fitParams", [](std::vector<double>& xdata_in,
                          std::vector<double>& ydata_in,
                          Gauss_Newton::GNParameters& gnParams,
                          py::function model,
                          std::map<std::string, double>& initialGuesses,
                          std::map<std::string, double>& extraParameters) {
        // Adaptar el modelo de Python a C++
        auto cpp_model = [&model](const std::vector<double>& x,
                                  const std::vector<double>& y, 
                                  const std::map<std::string, double>& params, 
                                  const std::map<std::string, double>& extraParams) {
          return model(x, y, params, extraParams).cast<std::vector<double>>();
        };

        // Llama a la implementación de C++
        return Gauss_Newton::fitParams(xdata_in, ydata_in, gnParams, cpp_model, initialGuesses, extraParameters);
    }, 
    py::arg("xdata_in"),
    py::arg("ydata_in"),
    py::arg("gnParams"),
    py::arg("model"),
    py::arg("initialGuesses"),
    py::arg("extraParameters"),
    "Fit parameters using the Gauss-Newton algorithm.");
}
