#pragma once
#include <iostream>
#include <map>
#include <functional>

namespace FittingAlgorithms{
  using StringDoubleMap = std::map<std::string, double>;

  template<class T2=double>
  using CostFunction    = std::function<double(const T2, const T2)>;

  template<class T1, class T2, class T3 = StringDoubleMap>
  using ModelFunction   = std::function<T2(const T1,
                                           const StringDoubleMap&,
                                           const T3&)>;
}
