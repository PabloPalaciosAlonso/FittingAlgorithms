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
