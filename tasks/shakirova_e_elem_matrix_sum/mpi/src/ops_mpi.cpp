#include "shakirova_e_elem_matrix_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "shakirova_e_elem_matrix_sum/common/include/common.hpp"
#include "shakirova_e_elem_matrix_sum/common/include/matrix.hpp"

namespace shakirova_e_elem_matrix_sum {

ShakirovaEElemMatrixSumMPI::ShakirovaEElemMatrixSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShakirovaEElemMatrixSumMPI::ValidationImpl() {
  return GetInput().rows > 0 && 
         GetInput().cols > 0 && 
         GetInput().data.size() == GetInput().rows * GetInput().cols;
}

bool ShakirovaEElemMatrixSumMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ShakirovaEElemMatrixSumMPI::RunImpl() {
  if (GetInput().rows == 0 || GetInput().cols == 0) {
    return false;
  }

  int rank = -1;
  int p_count = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p_count);

  size_t row_count = GetInput().rows;
  size_t col_count = GetInput().cols;

  size_t rows_chunk_size = row_count / p_count;
  size_t remainder_size = row_count % p_count;

  size_t start_row_index = (rows_chunk_size * rank) + std::min(static_cast<size_t>(rank), remainder_size);
  size_t end_row_index = start_row_index + rows_chunk_size + (std::cmp_less(rank, remainder_size) ? 1 : 0);

  int64_t partial_sum = 0;

  for (size_t i = start_row_index; i < end_row_index; i++) {
    for (size_t j = 0; j < col_count; j++) {
      partial_sum += GetInput().data[(i * col_count) + j];
    }
  }

  int64_t total_sum = 0;
  MPI_Allreduce(&partial_sum, &total_sum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  
  GetOutput() = total_sum;

  return true;
}

bool ShakirovaEElemMatrixSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shakirova_e_elem_matrix_sum