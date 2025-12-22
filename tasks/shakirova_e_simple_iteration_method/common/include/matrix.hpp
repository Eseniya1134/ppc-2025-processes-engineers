// matrix.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace shakirova_e_simple_iteration_method {

struct Matrix {
  size_t rows = 0;
  size_t cols = 0;
  std::vector<double> data;

  Matrix() = default;
  
  Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0) {}

  [[nodiscard]] bool IsValid() const {
    return rows > 0 && cols > 0 && data.size() == rows * cols;
  }

  [[nodiscard]] double& At(size_t i, size_t j) {
    return data[i * cols + j];
  }

  [[nodiscard]] const double& At(size_t i, size_t j) const {
    return data[i * cols + j];
  }

  friend bool operator==(const Matrix& v_left, const Matrix& v_right) {
    return v_left.rows == v_right.rows && 
           v_left.cols == v_right.cols && 
           v_left.data == v_right.data;
  }
};

}  // namespace shakirova_e_simple_iteration_method