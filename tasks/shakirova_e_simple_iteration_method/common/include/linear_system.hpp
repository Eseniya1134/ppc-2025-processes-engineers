#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "matrix.hpp"

namespace shakirova_e_simple_iteration_method {

struct LinearSystem {
  size_t n{0};
  Matrix A{};
  std::vector<double> b{};
  std::vector<double> x{};
  double epsilon{1e-6};
  size_t max_iterations{1000};

  LinearSystem() = default;

  explicit LinearSystem(size_t size) : n(size), A(size, size), b(size, 0.0), x(size, 0.0) {}

  LinearSystem(const Matrix &matrix, const std::vector<double> &rhs)
      : n(matrix.rows), A(matrix), b(rhs), x(matrix.rows, 0.0) {
    if (!matrix.IsValid() || matrix.rows != matrix.cols || matrix.rows != rhs.size()) {
      throw std::invalid_argument("Invalid matrix or vector dimensions");
    }
  }

  void SetSystem(const Matrix &matrix, const std::vector<double> &rhs) {
    if (!matrix.IsValid() || matrix.rows != matrix.cols || matrix.rows != rhs.size()) {
      throw std::invalid_argument("Invalid matrix or vector dimensions");
    }

    n = matrix.rows;
    A = matrix;
    b = rhs;
    x.assign(n, 0.0);
  }

  void SetInitialGuess(const std::vector<double> &initial_x) {
    if (initial_x.size() != n) {
      throw std::invalid_argument("Initial vector size does not match system size");
    }
    x = initial_x;
  }

  [[nodiscard]] bool IsValid() const noexcept {
    return n > 0 && A.IsValid() && A.rows == n && A.cols == n && b.size() == n && x.size() == n;
  }

  [[nodiscard]] bool HasNonZeroDiagonal() const noexcept {
    if (!IsValid()) {
      return false;
    }

    for (size_t i = 0; i < n; ++i) {
      if (std::abs(A.At(i, i)) <= 1e-12) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool HasDiagonalDominance() const noexcept {
    if (!IsValid()) {
      return false;
    }

    for (size_t i = 0; i < n; ++i) {
      const double diag = std::abs(A.At(i, i));
      double sum = 0.0;

      for (size_t j = 0; j < n; ++j) {
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

  // x = Bx + c
  [[nodiscard]] bool TransformToIterationForm(Matrix &B, std::vector<double> &c) const {
    if (!IsValid() || !HasNonZeroDiagonal()) {
      return false;
    }

    B = Matrix(n, n);
    c.assign(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
      const double aii = A.At(i, i);
      c[i] = b[i] / aii;

      for (size_t j = 0; j < n; ++j) {
        B.At(i, j) = (i == j) ? 0.0 : -A.At(i, j) / aii;
      }
    }

    return true;
  }

  [[nodiscard]] static double VectorNorm(const std::vector<double> &v) noexcept {
    double max_val = 0.0;
    for (double val : v) {
      max_val = std::max(max_val, std::abs(val));
    }
    return max_val;
  }

  [[nodiscard]] double MatrixNorm(const Matrix &M) const noexcept {
    double max_norm = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double row_sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        row_sum += std::abs(M.At(i, j));
      }
      max_norm = std::max(max_norm, row_sum);
    }

    return max_norm;
  }

  friend bool operator==(const LinearSystem &lhs, const LinearSystem &rhs) noexcept {
    if (lhs.n != rhs.n || !(lhs.A == rhs.A) || lhs.b.size() != rhs.b.size()) {
      return false;
    }

    for (size_t i = 0; i < lhs.b.size(); ++i) {
      if (std::abs(lhs.b[i] - rhs.b[i]) > 1e-10) {
        return false;
      }
    }

    return true;
  }
};

}  // namespace shakirova_e_simple_iteration_method
