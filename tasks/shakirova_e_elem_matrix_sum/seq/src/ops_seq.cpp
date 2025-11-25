#include "shakirova_e_elem_matrix_sum/seq/include/ops_seq.hpp"
#include "shakirova_e_elem_matrix_sum/common/include/common.hpp"
#include "shakirova_e_elem_matrix_sum/common/include/matrix.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace shakirova_e_elem_matrix_sum {

ShakirovaEElemMatrixSumSEQ::ShakirovaEElemMatrixSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShakirovaEElemMatrixSumSEQ::ValidationImpl() {
  return GetInput().rows > 0 &&
         GetInput().cols > 0 && 
         GetInput().data.size() == GetInput().cols * GetInput().rows;
}

bool ShakirovaEElemMatrixSumSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ShakirovaEElemMatrixSumSEQ::RunImpl() {
  if (GetInput().cols == 0 || GetInput().rows == 0) {
    return false;
  }

  GetOutput() = 0;

  for (size_t i = 0; i < GetInput().rows; i++) {
    for (size_t j = 0; j < GetInput().cols; j++) {
      GetOutput() += GetInput().at(i, j);
    }
  }

  return true;
}

bool ShakirovaEElemMatrixSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shakirova_e_elem_matrix_sum