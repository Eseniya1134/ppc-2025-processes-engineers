#include <gtest/gtest.h>

#include "shakirova_e_simple_iteration_method/common/include/common.hpp"
#include "shakirova_e_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "shakirova_e_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shakirova_e_simple_iteration_method {

class ShakirovaESimpleIterationMethodPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(ShakirovaESimpleIterationMethodPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShakirovaESimpleIterationMethodMPI, ShakirovaESimpleIterationMethodSEQ>(PPC_SETTINGS_example_processes_2);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShakirovaESimpleIterationMethodPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShakirovaESimpleIterationMethodPerfTest, kGtestValues, kPerfTestName);

}  // namespace shakirova_e_simple_iteration_method
