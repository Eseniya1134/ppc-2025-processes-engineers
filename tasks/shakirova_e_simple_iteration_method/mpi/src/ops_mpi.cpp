#include "shakirova_e_simple_iteration_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "shakirova_e_simple_iteration_method/common/include/common.hpp"
#include "shakirova_e_simple_iteration_method/common/include/linear_system.hpp"
#include "shakirova_e_simple_iteration_method/common/include/matrix.hpp"

namespace shakirova_e_simple_iteration_method {

namespace {

bool CheckInputValidity(const LinearSystem &input) {
  if (!input.IsValid()) {
    return false;
  }
  if (!input.HasNonZeroDiagonal()) {
    return false;
  }
  if (input.HasDiagonalDominance()) {
    return true;
  }

  Matrix b_matrix;
  std::vector<double> c_vector;
  if (!input.TransformToIterationForm(b_matrix, c_vector)) {
    return false;
  }
  return input.MatrixNorm(b_matrix) < 1.0;
}

struct DistributionParams {
  std::vector<int> rows_per_proc;
  std::vector<int> displacements;
  std::vector<int> matrix_elements;
  std::vector<int> matrix_offsets;
  int local_rows = 0;
};

DistributionParams CalculateScatterCounts(int world_size, int rank, size_t dimension) {
  DistributionParams params;
  params.rows_per_proc.resize(world_size);
  params.displacements.resize(world_size);
  params.matrix_elements.resize(world_size);
  params.matrix_offsets.resize(world_size);

  int base_rows = static_cast<int>(dimension) / world_size;
  int extra_rows = static_cast<int>(dimension) % world_size;
  int current_offset = 0;

  for (int proc = 0; proc < world_size; ++proc) {
    params.rows_per_proc[proc] = (proc < extra_rows) ? (base_rows + 1) : base_rows;
    params.displacements[proc] = current_offset;
    current_offset += params.rows_per_proc[proc];

    params.matrix_elements[proc] = params.rows_per_proc[proc] * static_cast<int>(dimension);
    params.matrix_offsets[proc] = params.displacements[proc] * static_cast<int>(dimension);
  }

  params.local_rows = params.rows_per_proc[rank];
  return params;
}

void ComputeLocalIteration(int local_rows, size_t dimension, const std::vector<double> &c_local,
                           const std::vector<double> &b_local, const std::vector<double> &x_current,
                           std::vector<double> &local_results) {
  for (int local_row = 0; local_row < local_rows; ++local_row) {
    double sum = c_local[local_row];
    size_t row_offset = static_cast<size_t>(local_row) * dimension;
    for (size_t col = 0; col < dimension; ++col) {
      sum += b_local[row_offset + col] * x_current[col];
    }
    local_results[local_row] = sum;
  }
}

}  // namespace

void ShakirovaESimpleIterationMethodMPI::SyncConfiguration(int rank, size_t &dimension, double &tolerance,
                                                           size_t &max_iter) {
  if (rank == 0) {
    const auto &input = GetInput();
    dimension = input.n;
    tolerance = input.epsilon;
    max_iter = input.max_iterations;
  }
  MPI_Bcast(&dimension, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&tolerance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&max_iter, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
}

bool ShakirovaESimpleIterationMethodMPI::PrepareSystemMatrices(int rank, size_t dimension, std::vector<double> &b_flat,
                                                               std::vector<double> &c_vector,
                                                               std::vector<double> &x_current) {
  int error = 0;
  if (rank == 0) {
    Matrix b_matrix;
    if (!GetInput().TransformToIterationForm(b_matrix, c_vector)) {
      error = 1;
    } else {
      b_flat = b_matrix.data;
      x_current = GetOutput();
    }
  }

  MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (error == 1) {
    return false;
  }

  if (rank != 0) {
    b_flat.resize(dimension * dimension);
    c_vector.resize(dimension);
    x_current.resize(dimension);
  }

  MPI_Bcast(b_flat.data(), static_cast<int>(dimension * dimension), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(c_vector.data(), static_cast<int>(dimension), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(x_current.data(), static_cast<int>(dimension), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool ShakirovaESimpleIterationMethodMPI::CheckConvergence(int rank, size_t dimension, double tolerance,
                                                          const std::vector<double> &x_next,
                                                          std::vector<double> &x_current) {
  bool converged = false;
  if (rank == 0) {
    double max_error = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
      max_error = std::max(max_error, std::abs(x_next[i] - x_current[i]));
    }
    converged = (max_error <= tolerance);
    x_current = x_next;
  }
  MPI_Bcast(&converged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return converged;
}

ShakirovaESimpleIterationMethodMPI::ShakirovaESimpleIterationMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ShakirovaESimpleIterationMethodMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  bool is_valid = true;

  if (world_rank == 0) {
    is_valid = CheckInputValidity(GetInput());
  }

  int valid_flag = is_valid ? 1 : 0;
  MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (valid_flag == 1);
}

bool ShakirovaESimpleIterationMethodMPI::PreProcessingImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    auto &input = GetInput();
    auto &output = GetOutput();

    size_t dimension = input.n;
    output.assign(dimension, 0.0);
  }

  return true;
}

bool ShakirovaESimpleIterationMethodMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  size_t dimension = 0;
  double tolerance_val = 0.0;
  size_t max_iterations = 0;

  SyncConfiguration(world_rank, dimension, tolerance_val, max_iterations);

  std::vector<double> b_flat;
  std::vector<double> c_vector;
  std::vector<double> x_current;

  if (!PrepareSystemMatrices(world_rank, dimension, b_flat, c_vector, x_current)) {
    return false;
  }

  DistributionParams dist = CalculateScatterCounts(world_size, world_rank, dimension);

  std::vector<double> b_local(static_cast<size_t>(dist.local_rows) * dimension);
  std::vector<double> c_local(dist.local_rows);
  std::vector<double> x_next(dimension);
  std::vector<double> local_results(dist.local_rows);

  MPI_Scatterv(b_flat.data(), dist.matrix_elements.data(), dist.matrix_offsets.data(), MPI_DOUBLE, b_local.data(),
               static_cast<int>(b_local.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(c_vector.data(), dist.rows_per_proc.data(), dist.displacements.data(), MPI_DOUBLE, c_local.data(),
               dist.local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  size_t iter_count = 0;
  bool converged = false;

  while (!converged && iter_count < max_iterations) {
    MPI_Bcast(x_current.data(), static_cast<int>(dimension), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    ComputeLocalIteration(dist.local_rows, dimension, c_local, b_local, x_current, local_results);

    MPI_Gatherv(local_results.data(), dist.local_rows, MPI_DOUBLE, x_next.data(), dist.rows_per_proc.data(),
                dist.displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    converged = CheckConvergence(world_rank, dimension, tolerance_val, x_next, x_current);
    iter_count++;
  }

  if (world_rank == 0) {
    if (!converged) {
      return false;
    }
    GetOutput() = x_current;
  }

  return true;
}

bool ShakirovaESimpleIterationMethodMPI::PostProcessingImpl() {
  int world_rank = 0;
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
