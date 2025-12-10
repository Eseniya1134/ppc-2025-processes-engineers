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


  


  
};

}  // namespace shakirova_e_simple_iteration_method
