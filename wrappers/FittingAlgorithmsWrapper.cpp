#include <pybind11/pybind11.h>
#include "GaussNewtonWrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(FittingAlgorithms, m) {
  // Descripción general del módulo
  m.doc() = "Fitting Algorithms module: includes multiple fitting algorithms.";
  
  // Registrar el submódulo Gauss-Newton
  auto gaussNewton = m.def_submodule("GaussNewton", "Gauss-Newton fitting algorithm");
  FittingAlgorithms::GaussNewton::registerGaussNewton(gaussNewton);
}
