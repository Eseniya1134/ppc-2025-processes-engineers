#include "shakirova_e_simple_iteration_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "shakirova_e_simple_iteration_method/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shakirova_e_simple_iteration_method {

ShakirovaESimpleIterationMethodMPI::ShakirovaESimpleIterationMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ShakirovaESimpleIterationMethodMPI::ValidationImpl() {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  auto& input = GetInput();

  if (!input.IsValid()) {
    return false;
  }

  bool has_nonzero_diag = input_.HasNonZeroDiagonal();
  if (!has_nonzero_diag) {
    return false;
  }
    
  bool has_dominance = input_.HasDiagonalDominance();
  if (!has_dominance) {
    Matrix B_matrix;
    std::vector<double> c_vector;
    bool transform_success = input_.TransformToIterationForm(B_matrix, c_vector);
      
    if (!transform_success) {
      return false;
    }
      
    double matrix_norm = input_.MatrixNorm(B_matrix);
    if (matrix_norm >= 1.0) {
      return false;
    }

  }  
  return true;
}

bool ShakirovaESimpleIterationMethodMPI::PreProcessingImpl() {
  auto& input = GetInput();
  auto& output = GetOutput();

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  if (world_rank == 0) {
    size_t dimension = input.n;
    output.resize(dimension, 0.0);
    
    output = input.x;
  }
  
  return true;
}

bool ShakirovaESimpleIterationMethodMPI::RunImpl() {

  auto& input = GetInput();
  auto& output = GetOutput();

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  size_t dimension = 0;
  double tolerance_val = 1e-6;
  
  if (world_rank == 0) {
    dimension = input.n;
    tolerance_val = input.epsilon;
  }
  
  MPI_Bcast(&dimension, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&tolerance_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  std::vector<double> B_flat;
  std::vector<double> c_vector;
  std::vector<double> x_current(dimension);
  
  if (world_rank == 0) {
    Matrix B_matrix;
    bool transform_ok = input.TransformToIterationForm(B_matrix, c_vector);
    
    if (!transform_ok) {
      return false;
    }
    
    B_flat = B_matrix.data;
    x_current = output;
  }
  
  if (world_rank == 0) {
    B_flat.resize(dimension * dimension);
    c_vector.resize(dimension);
  } else {
    B_flat.resize(dimension * dimension);
    c_vector.resize(dimension);
  }
  
  MPI_Bcast(B_flat.data(), dimension * dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(c_vector.data(), dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  std::vector<int> rows_per_proc(world_size);
  std::vector<int> displacements(world_size);
  
  int base_rows = dimension / world_size;
  int extra_rows = dimension % world_size;
  int current_offset = 0;
  
  for (int proc = 0; proc < world_size; ++proc) {
    rows_per_proc[proc] = (proc < extra_rows) ? (base_rows + 1) : base_rows;
    displacements[proc] = current_offset;
    current_offset += rows_per_proc[proc];
  }
  
  std::vector<int> matrix_elements_per_proc(world_size);
  std::vector<int> matrix_offsets(world_size);
  
  for (int proc = 0; proc < world_size; ++proc) {
    matrix_elements_per_proc[proc] = rows_per_proc[proc] * dimension;
    matrix_offsets[proc] = displacements[proc] * dimension;
  }
  
  int local_rows = rows_per_proc[world_rank];
  std::vector<double> B_local(local_rows * dimension, 0.0);
  std::vector<double> c_local(local_rows, 0.0);
  std::vector<double> x_next(dimension, 0.0);
  std::vector<double> local_results(local_rows, 0.0);
  
  MPI_Scatterv(B_flat.data(), matrix_elements_per_proc.data(), matrix_offsets.data(), 
               MPI_DOUBLE, B_local.data(), local_rows * dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  MPI_Scatterv(c_vector.data(), rows_per_proc.data(), displacements.data(), 
               MPI_DOUBLE, c_local.data(), local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  size_t iter_count = 0;
  size_t max_iterations = 1000;
  double convergence_error;
  
  if (world_rank == 0) {
    max_iterations = input.max_iterations;
  }
  MPI_Bcast(&max_iterations, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  
  do {
    MPI_Bcast(x_current.data(), dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (int local_row = 0; local_row < local_rows; ++local_row) {
      local_results[local_row] = c_local[local_row];
      
      int global_row = displacements[world_rank] + local_row;
      
      for (size_t col = 0; col < dimension; ++col) {
        if (static_cast<int>(col) != global_row) {
          local_results[local_row] += B_local[local_row * dimension + col] * x_current[col];
        }
      }
    }
    
    MPI_Gatherv(local_results.data(), local_rows, MPI_DOUBLE,
                x_next.data(), rows_per_proc.data(), displacements.data(), 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (world_rank == 0) {
      convergence_error = 0.0;
      for (size_t idx = 0; idx < dimension; ++idx) {
        double delta = std::abs(x_next[idx] - x_current[idx]);
        if (delta > convergence_error) {
          convergence_error = delta;
        }
      }
      x_current = x_next;
    }
    
    MPI_Bcast(&convergence_error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    iter_count++;
    
  } while (convergence_error > tolerance_val && iter_count < max_iterations);
  
  bool converged = iter_count < max_iterations;
  if (world_rank == 0) {
    if (!converged) {
      return false;
    }
    
    input.x = x_current;
    output = x_current;
  }
  
  return true;
}

bool ShakirovaESimpleIterationMethodMPI::PostProcessingImpl() {
  auto& input = GetInput();
  auto& output = GetOutput();

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  if (world_rank == 0) {
    const auto& x_solution = output;
    
    size_t dimension = input_.n;
    std::vector<double> residual_vector(dimension);
    
    for (size_t row = 0; row < dimension; row++) {
      residual_vector[row] = -input_.b[row];
      for (size_t col = 0; col < dimension; col++) {
        residual_vector[row] += input_.A.At(row, col) * x_solution[col];
      }
    }
    
    double norm_of_residual = LinearSystem::VectorNorm(residual_vector);
    double tolerance = input_.epsilon * 10;
    
    return norm_of_residual < tolerance;
  }
  
  return true;
}

}  // namespace shakirova_e_simple_iteration_method