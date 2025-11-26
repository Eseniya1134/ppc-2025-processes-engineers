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
    input_data_.rows = kSize_;
    input_data_.cols = kSize_;
    input_data_.data.assign(kSize_ * kSize_, 1);
    output_data_ = static_cast<int64_t>(kSize_) * static_cast<int64_t>(kSize_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  const size_t kSize_ = 8000;
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