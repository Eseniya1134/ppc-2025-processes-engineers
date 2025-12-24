#include <gtest/gtest.h>
#include <mpi.h>

#include <random>
#include <vector>

#include "shakirova_e_shells_sort_simple/common/include/common.hpp"
#include "shakirova_e_shells_sort_simple/common/include/shell_sort.hpp"
#include "shakirova_e_shells_sort_simple/mpi/include/ops_mpi.hpp"
#include "shakirova_e_shells_sort_simple/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shakirova_e_shells_sort_simple {

class ShakirovaEShellsSortSimplePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int count = 1500000;
    input_data_.resize(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-150000, 150000);

    for (int &val : input_data_) {
      val = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return true;
    }

    std::vector<int> reference_data = input_data_;
    if (!reference_data.empty()) {
      ShellSortImpl(reference_data, 0, static_cast<int>(reference_data.size()) - 1);
    }

    return output_data == reference_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(ShakirovaEShellsSortSimplePerfTests, ExecutePerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShakirovaEShellsSortSimpleMPI, ShakirovaEShellsSortSimpleSEQ>(
        PPC_SETTINGS_shakirova_e_shells_sort_simple);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = ShakirovaEShellsSortSimplePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ShellSortPerformanceTests, ShakirovaEShellsSortSimplePerfTests, kGtestValues, kPerfTestName);

}  // namespace shakirova_e_shells_sort_simple
