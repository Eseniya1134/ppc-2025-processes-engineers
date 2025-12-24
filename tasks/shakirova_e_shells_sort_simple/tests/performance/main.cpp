#include <gtest/gtest.h>

#include "shakirova_e_shells_sort_simple/common/include/common.hpp"
#include "shakirova_e_shells_sort_simple/mpi/include/ops_mpi.hpp"
#include "shakirova_e_shells_sort_simple/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shakirova_e_shells_sort_simple {

class ShakirovaEShellsSortSimplePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(ShakirovaEShellsSortSimplePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShakirovaEShellsSortSimpleMPI, ShakirovaEShellsSortSimpleSEQ>(
        PPC_SETTINGS_example_processes_3);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShakirovaEShellsSortSimplePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShakirovaEShellsSortSimplePerfTests, kGtestValues, kPerfTestName);

}  // namespace shakirova_e_shells_sort_simple
