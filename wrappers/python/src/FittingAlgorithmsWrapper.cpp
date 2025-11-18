#include <pybind11/pybind11.h>
#include "registerDefinitions.h"
#include "utils/costFunctionsWrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(FittingAlgorithms, m) {
  // Descripción general del módulo
  m.doc() = "Fitting Algorithms module: includes multiple fitting algorithms.";
  
  // Registrar el submódulo Gauss-Newton
  auto gaussNewton       = m.def_submodule("GaussNewton",       "Gauss-Newton fitting algorithm");
  auto parallelTempering = m.def_submodule("ParallelTempering", "Parallel Tempering fitting algorithm");
  FittingAlgorithms::GaussNewton::registerGaussNewton(gaussNewton);
  FittingAlgorithms::ParallelTempering::registerParallelTempering(parallelTempering);
  FittingAlgorithms::registerCostFunctions(m);
}
