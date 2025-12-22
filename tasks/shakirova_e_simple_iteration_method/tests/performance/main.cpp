#include <gtest/gtest.h>

#include "shakirova_e_simple_iteration_method/common/include/common.hpp"
#include "shakirova_e_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "shakirova_e_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shakirova_e_simple_iteration_method {

using InType  = LinearSystem;
using OutType = std::vector<double>;

static LinearSystem GenerateTestSystem(size_t n, double dominance_factor = 2.0) {
  Matrix A(n, n);
  std::vector<double> b(n);

  for (size_t i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      if (i != j) {
        A.At(i, j) = 1.0;
        row_sum += 1.0;
      }
    }
    A.At(i, i) = row_sum * dominance_factor;
    b[i] = A.At(i, i) + (n - 1);
  }

  LinearSystem system(A, b);
  system.epsilon = 1e-6;
  system.max_iterations = 1000;

  return system;
}

class ShakirovaESimpleIterationMethodPerfTest
    : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  size_t matrix_size_ = 500;

  void SetUp() override {
    matrix_size_ = 500;
  }

  InType GetTestInputData() override {
    return GenerateTestSystem(matrix_size_);
  }

  bool CheckTestOutputData(OutType& output_data) override {
    return output_data.size() == matrix_size_;
  }
};

TEST_P(ShakirovaESimpleIterationMethodPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<
        InType,
        ShakirovaESimpleIterationMethodMPI,
        ShakirovaESimpleIterationMethodSEQ>(
        PPC_SETTINGS_example_processes_2);

const auto kGtestValues =
    ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName =
    ShakirovaESimpleIterationMethodPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(
    RunModeTests,
    ShakirovaESimpleIterationMethodPerfTest,
    kGtestValues,
    kPerfTestName);

}  // namespace shakirova_e_simple_iteration_method