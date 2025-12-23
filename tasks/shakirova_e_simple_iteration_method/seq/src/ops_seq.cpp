#include "shakirova_e_simple_iteration_method/seq/include/ops_seq.hpp"
#include <cmath>
#include <algorithm>

#include "shakirova_e_simple_iteration_method/seq/include/ops_seq.hpp"
#include <cmath>
#include <algorithm>

namespace shakirova_e_simple_iteration_method {

ShakirovaESimpleIterationMethodSEQ::ShakirovaESimpleIterationMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ShakirovaESimpleIterationMethodSEQ::ValidationImpl() {
  auto& input = GetInput();

  if (!input.IsValid()) {
    return false;
  }
  
  bool has_nonzero_diag = input.HasNonZeroDiagonal();
  if (!has_nonzero_diag) {
    return false;
  }
  
  bool has_dominance = input.HasDiagonalDominance();
  if (!has_dominance) {
    Matrix B_matrix;
    std::vector<double> c_vector;
    bool transform_success = input.TransformToIterationForm(B_matrix, c_vector);
    
    
    if (!transform_success) {
      return false;
    }
    
    double matrix_norm = input.MatrixNorm(B_matrix);
    if (matrix_norm >= 1.0) {
      return false; 
    }
  }
  
  return true;
}

bool ShakirovaESimpleIterationMethodSEQ::PreProcessingImpl() {
  auto& input = GetInput();
  auto& output = GetOutput();

  size_t dimension = input.n;
  output.resize(dimension, 0.0);
  
  output.assign(input.n, 0.0); 
  
  return true;
}

bool ShakirovaESimpleIterationMethodSEQ::RunImpl() {
  auto& input = GetInput();
  auto& output = GetOutput();

  auto& x_current = output;
  
  Matrix B_matrix;
  std::vector<double> c_vector;
  output.assign(input.n, 0.0);

  bool transform_ok = input.TransformToIterationForm(B_matrix, c_vector);
  if (!transform_ok) return false;

  double B_norm = input.MatrixNorm(B_matrix);
  if (B_norm == 0.0) {
    for (size_t i = 0; i < input.n; ++i) {
      output[i] = c_vector[i];
    }
    return true;
  }
  
  if (!transform_ok) {
    return false;
  }
  
  size_t dimension = input.n;
  std::vector<double> x_next(dimension);
  size_t iter_count = 0;
  double convergence_error;
  
  do {
    for (size_t row = 0; row < dimension; row++) {
      x_next[row] = c_vector[row];
      for (size_t col = 0; col < dimension; col++) {
        x_next[row] += B_matrix.At(row, col) * x_current[col];
      }
    }
    
    std::vector<double> difference(dimension);
    for (size_t idx = 0; idx < dimension; idx++) {
      difference[idx] = x_next[idx] - x_current[idx];
    }
    
    convergence_error = LinearSystem::VectorNorm(difference);
    
    x_current = x_next;
    iter_count++;
    
  } while (convergence_error > input.epsilon && iter_count < input.max_iterations);
  
  if (convergence_error > input.epsilon) return false;
  
  return true;
}

bool ShakirovaESimpleIterationMethodSEQ::PostProcessingImpl() {
  auto& input = GetInput();
  auto& output = GetOutput();

  const auto& x_solution = output;
  
  size_t dimension = input.n;
  std::vector<double> residual_vector(dimension);
  
  for (size_t row = 0; row < dimension; row++) {
    residual_vector[row] = -input.b[row];
    for (size_t col = 0; col < dimension; col++) {
      residual_vector[row] += input.A.At(row, col) * x_solution[col];
    }
  }
  
  double norm_of_residual = LinearSystem::VectorNorm(residual_vector);
  double tolerance = input.epsilon * 10;
  
  return norm_of_residual < tolerance;
}

}  // namespace shakirova_e_simple_iteration_method