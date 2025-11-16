#include <gtest/gtest.h>

#include "shakirova_e_elem_matrix_sum/common/include/common.hpp"
#include "shakirova_e_elem_matrix_sum/mpi/include/ops_mpi.hpp"
#include "shakirova_e_elem_matrix_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shakirova_e_elem_matrix_sum {

class ShakirovaEElemMatrixSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ShakirovaEElemMatrixSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShakirovaEElemMatrixSumMPI, ShakirovaEElemMatrixSumSEQ>(PPC_SETTINGS_shakirova_e_elem_matrix_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShakirovaEElemMatrixSumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShakirovaEElemMatrixSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace shakirova_e_elem_matrix_sum
