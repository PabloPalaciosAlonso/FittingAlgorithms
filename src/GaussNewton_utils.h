#include <vector>
#include <map>
#include <string>
#include "defines.h"

namespace FittingAlgorithms{
  namespace GaussNewton{
    
    inline std::vector<double> mapToSTDVector(const StringDoubleMap& initialGuesses){
      std::size_t total_size = initialGuesses.size();
      std::vector<double> v(total_size);
      
      std::size_t index = 0;
      for (const auto& pair : initialGuesses)
        v[index++] = pair.second;

      return v;
    }

    inline void updateMapFromSTDVector(StringDoubleMap& m,
                                       const std::vector<double>& v){
      if (m.size() != v.size()) {
        std::cerr << "Error: The size of the vector (" << v.size()
                  << ") does not match the size of the map (" << m.size() << ")"
                  << std::endl;
        return;
      }
      
      auto it = m.begin();
      for (std::size_t i = 0; i < v.size(); ++i) {
        it->second = v[i];
        ++it;
      }
    }

    inline void addStdVectors(std::vector<double>& a,
                              const std::vector<double>& b)
    {
      if (a.size() != b.size()) {
        std::cerr << "Error: vector sizes differ (" << a.size()
                  << " vs " << b.size() << ")" << std::endl;
        return;
      }
      
      std::transform(a.begin(), a.end(),
                     b.begin(),
                     a.begin(),
                     [](double x, double y) { return x + y; });
    }
    
    inline double computeRelativeNorm(const std::vector<double>& delta,
                                      const std::vector<double>& fitting,
                                      double n){
      
      if (delta.size() != fitting.size()) {
        std::cerr << "Error: size mismatch (" << delta.size()
                  << " vs " << fitting.size() << ")\n";
        return 0.0;
      }
      
      double sum = 0.0;
      
      for (std::size_t i = 0; i < delta.size(); ++i) {
        double ratio = delta[i] / fitting[i];
        sum += ratio * ratio;
      }
      return std::sqrt(sum) / n;
    }
    
    inline void printParameters(const StringDoubleMap& parameters, int iter) {
      std::cout<<"Iteration "<<iter<<std::endl;
      for (const auto& param : parameters) {
        std::cout << "  " << param.first << " = " << param.second << std::endl;
      }
      std::cout<<"\n";
    }
  }
}
