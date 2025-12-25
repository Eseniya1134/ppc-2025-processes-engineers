#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "matrix.hpp"

namespace shakirova_e_simple_iteration_method {
struct LinearSystem {
  size_t n{};
  Matrix A;
  std::vector<double> b;
  std::vector<double> x;
  double epsilon = 1e-6;
  size_t max_iterations = 1000;

  LinearSystem() {}

  explicit LinearSystem(size_t size) : n(size) {
    A.rows = size;
    A.cols = size;
    A.data.resize(size * size, 0.0);
    b.resize(size, 0.0);
    x.resize(size, 0.0);
  }

  LinearSystem(const Matrix &matrix, const std::vector<double> &rhs) {
    if (!matrix.IsValid() || matrix.rows != matrix.cols || matrix.rows != rhs.size()) {
      throw std::invalid_argument("Invalid matrix or vector dimensions");
    }
    n = matrix.rows;
    A = matrix;
    b = rhs;
    x.resize(n, 0.0);
  }

  void SetSystem(const Matrix &matrix, const std::vector<double> &rhs) {
    if (!matrix.IsValid() || matrix.rows != matrix.cols || matrix.rows != rhs.size()) {
      throw std::invalid_argument("Invalid matrix or vector dimensions");
    }
    n = matrix.rows;
    A = matrix;
    b = rhs;
    x.resize(n, 0.0);
  }

  void SetInitialGuess(const std::vector<double> &initial_x) {
    if (initial_x.size() != n) {
      throw std::invalid_argument("The initial size does not match the system size.");
    }
    x = initial_x;
  }

  [[nodiscard]] bool IsValid() const {
    return (n > 0 && n == b.size() && n == x.size() && A.data.size() == n * n && A.rows == n && A.cols == n);
  }

  [[nodiscard]] bool HasNonZeroDiagonal() const {
    for (size_t i = 0; i < n; i++) {
      if (std::abs(A.At(i, i)) < 1e-12) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool HasDiagonalDominance() const {
    if (!IsValid()) {
      return false;
    }
    for (size_t i = 0; i < n; i++) {
      double diag = std::abs(A.At(i, i));
      double sum = 0.0;
      for (size_t j = 0; j < n; j++) {
        if (i != j) {
          sum += std::abs(A.At(i, j));
        }
      }
      if (diag <= sum + 1e-12) {
        return false;
      }
    }
    return true;
  }

  // Преобразование к виду x = Bx + c
  [[nodiscard]] bool TransformToIterationForm(Matrix &b_matrix, std::vector<double> &c_vector) const {
    if (!IsValid() || !HasNonZeroDiagonal()) {
      return false;
    }
    b_matrix = Matrix(n, n);
    c_vector.assign(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
      const double aii = A.At(i, i);
      c_vector[i] = b[i] / aii;

      for (size_t j = 0; j < n; ++j) {
        b_matrix.At(i, j) = (i == j) ? 0.0 : -A.At(i, j) / aii;
      }
    }
    return true;
  }

  // Норма вектора (максимальная норма)
  [[nodiscard]] static double VectorNorm(const std::vector<double> &v) {
    double max_val = 0.0;
    for (double val : v) {
      max_val = std::max(max_val, std::abs(val));
    }
    return max_val;
  }

  // Норма матрицы
  [[nodiscard]] double MatrixNorm(const Matrix &matrix) const noexcept {
    double max_norm = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double row_sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        row_sum += std::abs(matrix.At(i, j));
      }
      max_norm = std::max(max_norm, row_sum);
    }
    return max_norm;
  }

  friend bool operator==(const LinearSystem &lhs, const LinearSystem &rhs) {
    if (lhs.n != rhs.n) {
      return false;
    }
    if (!(lhs.A == rhs.A)) {
      return false;
    }
    if (lhs.b.size() != rhs.b.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs.b.size(); i++) {
      if (std::abs(lhs.b[i] - rhs.b[i]) > 1e-10) {
        return false;
      }
    }
    return true;
  }
};
}  // namespace shakirova_e_simple_iteration_method
