#pragma once
#include <iostream>
#include <map>
#include <functional>

#define fori(x,y) for(int i=x; i<int(y); i++)
#define forj(x,y) for(int j=x; j<int(y); j++)

namespace FittingAlgorithms{
  using StringDoubleMap = std::map<std::string, double>;
  using CostFunction    = std::function<double(const double, const double)>;
  
  template <typename T>
  using ModelFunction = std::function<double(const T,
                                             const StringDoubleMap&,
                                             const StringDoubleMap&)>;
}
