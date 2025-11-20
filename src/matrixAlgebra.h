#pragma once
#include <vector>
#include <map>
#include <string>
#include "defines.h"

#if defined(USE_EIGEN)

#include <Eigen/Dense>

namespace FittingAlgorithms{
  using vector = Eigen::VectorXd;
  using matrix = Eigen::MatrixXd;
  
  inline matrix transpose(const matrix &M){
    return M.transpose();
  }

  inline matrix matrixProduct(const matrix &M1, const matrix &M2){
    return M1 * M2;
  }

  inline vector matrixProduct(const matrix &M1, const vector &M2){
    return M1 * M2;
  }

  inline matrix solve(const matrix &A, const matrix &B){
    Eigen::LDLT<matrix> ldlt(A);
    return ldlt.solve(B);
  }

  inline matrix inverse(const matrix &M){
    return M.inverse();
  }

  inline double squaredNorm(const vector &v){
    return v.squaredNorm();    
  }
}

#else

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>

namespace FittingAlgorithms{
  namespace ublas = boost::numeric::ublas;
  using vector = ublas::vector<double>;
  using matrix = ublas::matrix<double>;
  
  inline matrix transpose(const matrix &M){
    return ublas::trans(M);
  }

  inline matrix matrixProduct(const matrix &M1, const matrix &M2){
    return ublas::prod(M1, M2);
  }

  inline vector matrixProduct(const matrix &M1, const vector &M2){
    return ublas::prod(M1, M2);
  }

  inline matrix solve(const matrix &M1, const matrix &M2){
    const std::size_t n = M1.size1();
    const std::size_t m = M2.size2();

    matrix A = M1;  // copy because LU overwrites
    matrix X = M2;

    ublas::permutation_matrix<std::size_t> pm(n);

    int res = ublas::lu_factorize(A, pm);
    if(res != 0)
      throw std::runtime_error("LU factorization failed");

    ublas::lu_substitute(A, pm, X);
    return X;
  }

  inline matrix inverse(const matrix &M1){
    if(M1.size1() != M1.size2())
      throw std::runtime_error("inverse: matrix must be square");

    const std::size_t n = M1.size1();
    
    ublas::identity_matrix<double> I(n);
    
    return solve(M1, I);
  }

  inline double squaredNorm(const vector &v){
    return ublas::inner_prod(v, v);
  }
}
#endif












 //  inline vector mapToEigen(const StringDoubleMap& initialGuesses) {
    
//     std::size_t total_size = initialGuesses.size();
    
//     // Create an Eigen::VectorXd with the size of the map
//     vector eigenVec(total_size);
    
//     // Populate the Eigen::VectorXd with values from the map
//     std::size_t index = 0;
//     for (const auto& pair : initialGuesses) {
//       eigenVec(index++) = pair.second; // Store the map value in the vector
//     }
    
//     return eigenVec;
//   }
  
//   // Function to update a StringDoubleMap with values from an Eigen::VectorXd
//   inline void updateMapFromEigen(StringDoubleMap& m, const vector& eigenVec) {
//     // Ensure that the size of the Eigen::VectorXd matches the size of the std::map
//     if (m.size() != int(eigenVec.size())) {
//       std::cerr << "Error: The size of the vector (" << eigenVec.size()
//                 << ") does not match the size of the map (" << m.size() << ")" << std::endl;
//       return;
//     }
    
//     // Iterator to traverse the map
//     auto it = m.begin();
    
//     // Iterate through the Eigen::VectorXd values and update the map
//     for (int i = 0; i < eigenVec.size(); ++i) {
//       it->second = eigenVec(i);  // Update the value in the map
//       ++it;  // Move to the next element in the map
//     }
//   }

//   inline int index2D(int row, int col, int numberCols){
//     return col + row*numberCols;    
//   }
  
//   inline Eigen::VectorXd eigenVectorFromStd(const std::vector<double> &v) {
//     return Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
//   }

//   inline Eigen::MatrixXd eigenMatrixFromStd(const std::vector<double> &v,
//                                             int numberRows, int numberCols){
//     Eigen::MatrixXd M(numberRows, numberCols);
//     for (int r = 0; r < numberRows; ++r)
//       for (int c = 0; c < numberCols; ++c)
//         M(r, c) = v[index2D(r,c,numberCols)];  
//     return M;
//   }

//   inline std::vector<double> stdFromEigenVector(const Eigen::VectorXd &v) {
//     return std::vector<double>(v.data(), v.data() + v.size());
// }

//   inline std::vector<double> stdFromEigenMatrix(const Eigen::MatrixXd &M) {
//     int rows = M.rows();
//     int cols = M.cols();
    
//     std::vector<double> v(rows * cols);

//     for (int r = 0; r < rows; ++r)
//         for (int c = 0; c < cols; ++c)
//             v[index2D(r, c, cols)] = M(r, c);

//     return v;
// }


  
  
// }
