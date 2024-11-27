#pragma once

#include <pybind11/pybind11.h>

namespace FittingAlgorithms {
  namespace GaussNewton {    
    extern void registerGaussNewton(pybind11::module_& parent_module);
  }
}
