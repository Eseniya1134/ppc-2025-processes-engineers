#include "shakirova_e_simple_iteration_method/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "shakirova_e_simple_iteration_method/common/include/common.hpp"
#include "shakirova_e_simple_iteration_method/common/include/linear_system.hpp"
#include "shakirova_e_simple_iteration_method/common/include/matrix.hpp"

namespace shakirova_e_simple_iteration_method {

ShakirovaESimpleIterationMethodSEQ::ShakirovaESimpleIterationMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ShakirovaESimpleIterationMethodSEQ::ValidationImpl() {
  auto &input = GetInput();

  if (!input.IsValid()) {
    return false;
  }

  bool has_nonzero_diag = input.HasNonZeroDiagonal();
  if (!has_nonzero_diag) {
    return false;
  }

  bool has_dominance = input.HasDiagonalDominance();
  if (!has_dominance) {
    Matrix bMatrix;
    std::vector<double> cVector;
    bool transformSuccess = input.TransformToIterationForm(bMatrix, cVector);

    if (!transformSuccess) {
      return false;
    }

    double matrixNorm = input.MatrixNorm(bMatrix);
    if (matrixNorm >= 1.0) {
      return false;
    }
  }

  return true;
}

bool ShakirovaESimpleIterationMethodSEQ::PreProcessingImpl() {
  auto &input = GetInput();
  auto &output = GetOutput();

  size_t dimension = input.n;
  output.assign(dimension, 0.0);

  return true;
}

bool ShakirovaESimpleIterationMethodSEQ::RunImpl() {
  auto &input = GetInput();
  auto &output = GetOutput();
  auto &xCurrent = output;

  Matrix bMatrix;
  std::vector<double> cVector;
  output.assign(input.n, 0.0);

  bool transformOk = input.TransformToIterationForm(bMatrix, cVector);
  if (!transformOk) {
    return false;
  }

  double bNorm = input.MatrixNorm(bMatrix);
  if (bNorm == 0.0) {
    for (size_t i = 0; i < input.n; ++i) {
      output[i] = cVector[i];
    }
    return true;
  }

  size_t dimension = input.n;
  std::vector<double> xNext(dimension, 0.0);
  size_t iterCount = 0;
  double convergenceError = NAN;

  while (true) {
    for (size_t row = 0; row < dimension; ++row) {
      xNext[row] = cVector[row];
      for (size_t col = 0; col < dimension; ++col) {
        xNext[row] += bMatrix.At(row, col) * xCurrent[col];
      }
    }

    std::vector<double> difference(dimension, 0.0);
    for (size_t idx = 0; idx < dimension; ++idx) {
      difference[idx] = xNext[idx] - xCurrent[idx];
    }

    convergenceError = LinearSystem::VectorNorm(difference);

    xCurrent = xNext;
    iterCount++;

    if (convergenceError <= input.epsilon || iterCount >= input.max_iterations) {
      break;
    }
  }

  return convergenceError <= input.epsilon;
}

bool ShakirovaESimpleIterationMethodSEQ::PostProcessingImpl() {
  auto &input = GetInput();
  auto &output = GetOutput();
  const auto &xSolution = output;

  size_t dimension = input.n;
  std::vector<double> residualVector(dimension, 0.0);

  for (size_t row = 0; row < dimension; ++row) {
    residualVector[row] = -input.b[row];
    for (size_t col = 0; col < dimension; ++col) {
      residualVector[row] += input.A.At(row, col) * xSolution[col];
    }
  }

  double normOfResidual = LinearSystem::VectorNorm(residualVector);
  double tolerance = input.epsilon * 10.0;

  return normOfResidual < tolerance;
}

}  // namespace shakirova_e_simple_iteration_method
