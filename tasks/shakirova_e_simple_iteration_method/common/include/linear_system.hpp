#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include "matrix.hpp"

namespace shakirova_e_simple_iteration_method {

struct LinearSystem {
  size_t n;                         // Размерность системы
  Matrix A;                         // Матрица в одномерном массиве
  std::vector<double> b;            // Вектор правых частей
  std::vector<double> x;            // Вектор решения
  double epsilon = 1e-6;        // Точность
  size_t max_iterations = 1000; // Максимальное число итераций

  // Конструктор
  LinearSystem(size_t size) : n(size) {
    A.rows = size;
    A.cols = size;
    A.data.resize(size * size, 0);
    b.resize(size, 0.0);
    x.resize(size, 0.0);  // Начальное приближение
  }
  
  [[nodiscard]] bool IsValid() const {
    return (n > 0 && 
        n == b.size() && 
        n == x.size() && 
        A.data.size() == n * n &&
        A.rows == n && 
        A.cols == n);
  }

  [[nodiscard]] bool HasNonZeroDiagonal() const {
    for (int i = 0; i < n; i++){
        if (A.At(i, i) == 0) return false;
    } return true;
  }

  [[nodiscard]] bool HasDiagonalDominance() const {
    if (!IsValid()) return false;
    
    for (size_t i = 0; i < n; i++) {
      double diag = std::abs(static_cast<double>(A.At(i, i)));
      double sum = 0.0;
      
      for (size_t j = 0; j < n; j++) {
        if (i != j) {
          sum += std::abs(static_cast<double>(A.At(i, j)));
        }
      }
      
      if (diag <= sum + 1e-12) {
        return false;
      }
    }
    return true;
  }
  
   // Преобразование к виду x = Bx + c
  [[nodiscard]] bool TransformToIterationForm(Matrix& B, std::vector<double>& c) const {
    if (!IsValid() || !HasNonZeroDiagonal()) return false;
    
    B.rows = n;
    B.cols = n;
    B.data.resize(n * n);
    c.resize(n);
    
    for (size_t i = 0; i < n; i++) {
      double aii = A.At(i, i);
      c[i] = b[i] / aii;
      
      for (size_t j = 0; j < n; j++) {
        if (i == j) {
          B.At(i, j) = 0.0;
        } else {
          B.At(i, j) = -A.At(i, j) / aii;
        }
      }
    }
    return true;
  }

  // норма вектора
  [[nodiscard]] static double VectorNorm(const std::vector<double>& v) {
    double max_val = 0.0;
    for (double val : v) {
      max_val = std::max(max_val, std::abs(val));
    }
    return max_val;
  }
  
  // норма матрицы
  [[nodiscard]] double MatrixNorm(const Matrix& M) const {
    double max_norm = 0.0;
    for (size_t i = 0; i < n; i++) {
      double row_sum = 0.0;
      for (size_t j = 0; j < n; j++) {
        row_sum += std::abs(M.At(i, j));
      }
      max_norm = std::max(max_norm, row_sum);
    }
    return max_norm;
  }


  
};

}  // namespace shakirova_e_simple_iteration_method
