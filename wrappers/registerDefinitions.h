#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace FittingAlgorithms {
  namespace GaussNewton {    
    extern void registerGaussNewton(pybind11::module_& parent_module);
  }
  namespace ParallelTempering {    
    extern void registerParallelTempering(pybind11::module_& parent_module);
  }
}
