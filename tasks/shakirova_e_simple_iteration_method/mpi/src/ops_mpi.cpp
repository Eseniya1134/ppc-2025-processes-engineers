#include "shakirova_e_simple_iteration_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "shakirova_e_simple_iteration_method/common/include/common.hpp"

namespace shakirova_e_simple_iteration_method {

ShakirovaESimpleIterationMethodMPI::ShakirovaESimpleIterationMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ShakirovaESimpleIterationMethodMPI::ValidationImpl() {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  bool is_valid = true;

  if (world_rank == 0) {
    auto &input = GetInput();

    if (!input.IsValid()) {
      is_valid = false;
    } else {
      bool has_nonzero_diag = input.HasNonZeroDiagonal();
      if (!has_nonzero_diag) {
        is_valid = false;
      } else {
        bool has_dominance = input.HasDiagonalDominance();
        if (!has_dominance) {
          Matrix B_matrix;
          std::vector<double> c_vector;
          bool transform_success = input.TransformToIterationForm(B_matrix, c_vector);

          if (!transform_success) {
            is_valid = false;
          } else {
            double matrix_norm = input.MatrixNorm(B_matrix);
            if (matrix_norm >= 1.0) {
              is_valid = false;
            }
          }
        }
      }
    }
  }

  int valid_flag = is_valid ? 1 : 0;
  MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  is_valid = (valid_flag == 1);

  return is_valid;
}

bool ShakirovaESimpleIterationMethodMPI::PreProcessingImpl() {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    auto &input = GetInput();
    auto &output = GetOutput();

    size_t dimension = input.n;
    output.resize(dimension, 0.0);
    output.assign(input.n, 0.0);
  }

  return true;
}

bool ShakirovaESimpleIterationMethodMPI::RunImpl() {
  auto &input = GetInput();
  auto &output = GetOutput();

  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  size_t dimension = 0;
  double tolerance_val = 1e-6;
  size_t max_iterations = 1000;

  if (world_rank == 0) {
    dimension = input.n;
    tolerance_val = input.epsilon;
    max_iterations = input.max_iterations;
  }

  MPI_Bcast(&dimension, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&tolerance_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&max_iterations, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  std::vector<double> B_flat(dimension * dimension);
  std::vector<double> c_vector(dimension);
  std::vector<double> x_current(dimension);

  if (world_rank == 0) {
    Matrix B_matrix;
    bool transform_ok = input.TransformToIterationForm(B_matrix, c_vector);

    if (!transform_ok) {
      int error_flag = 0;
      MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return false;
    }

    int error_flag = 1;
    MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    B_flat = B_matrix.data;
    x_current = output;
  } else {
    int error_flag;
    MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (error_flag == 0) {
      return false;
    }
  }

  MPI_Bcast(B_flat.data(), dimension * dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(c_vector.data(), dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(x_current.data(), dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

  MPI_Scatterv(B_flat.data(), matrix_elements_per_proc.data(), matrix_offsets.data(), MPI_DOUBLE, B_local.data(),
               local_rows * dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(c_vector.data(), rows_per_proc.data(), displacements.data(), MPI_DOUBLE, c_local.data(), local_rows,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  size_t iter_count = 0;
  double convergence_error = 0.0;

  do {
    MPI_Bcast(x_current.data(), dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int local_row = 0; local_row < local_rows; ++local_row) {
      local_results[local_row] = c_local[local_row];

      for (size_t col = 0; col < dimension; ++col) {
        local_results[local_row] += B_local[local_row * dimension + col] * x_current[col];
      }
    }

    MPI_Gatherv(local_results.data(), local_rows, MPI_DOUBLE, x_next.data(), rows_per_proc.data(), displacements.data(),
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

  if (world_rank == 0) {
    bool converged = convergence_error <= tolerance_val;
    if (!converged) {
      return false;
    }

    output = x_current;
  }

  return true;
}

bool ShakirovaESimpleIterationMethodMPI::PostProcessingImpl() {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    auto &input = GetInput();
    auto &output = GetOutput();

    const auto &x_solution = output;
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

  return true;
}

}  // namespace shakirova_e_simple_iteration_method
