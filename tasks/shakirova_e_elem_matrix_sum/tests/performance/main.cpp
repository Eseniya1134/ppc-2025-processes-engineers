#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "shakirova_e_elem_matrix_sum/common/include/common.hpp"
#include "shakirova_e_elem_matrix_sum/common/include/matrix.hpp"
#include "shakirova_e_elem_matrix_sum/mpi/include/ops_mpi.hpp"
#include "shakirova_e_elem_matrix_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shakirova_e_elem_matrix_sum {

class ShakirovaEElemMatrixSumPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_.rows = matrix_size;
    input_data_.cols = matrix_size;
    input_data_.data.resize(matrix_size * matrix_size);

    for (size_t i = 0; i < input_data_.data.size(); i++) {
      input_data_.data[i] = 1;
    }
    
    output_data_ = static_cast<int64_t>(matrix_size) * static_cast<int64_t>(matrix_size);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  const size_t matrix_size = 16000;
  InType input_data_ = {};
  OutType output_data_ = 0;
};

TEST_P(ShakirovaEElemMatrixSumPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShakirovaEElemMatrixSumMPI, ShakirovaEElemMatrixSumSEQ>(
        PPC_SETTINGS_shakirova_e_elem_matrix_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShakirovaEElemMatrixSumPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShakirovaEElemMatrixSumPerfTest, kGtestValues, kPerfTestName);

}  // namespace shakirova_e_elem_matrix_sum