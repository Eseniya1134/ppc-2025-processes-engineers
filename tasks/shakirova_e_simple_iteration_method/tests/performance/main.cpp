#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "shakirova_e_simple_iteration_method/common/include/common.hpp"
#include "shakirova_e_simple_iteration_method/common/include/linear_system.hpp"
#include "shakirova_e_simple_iteration_method/common/include/matrix.hpp"
#include "shakirova_e_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "shakirova_e_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shakirova_e_simple_iteration_method {

using InType = LinearSystem;
using OutType = std::vector<double>;

static LinearSystem GenerateTestSystem(size_t n) {
  Matrix mat(n, n);
  std::vector<double> b(n);

  for (size_t i = 0; i < n; ++i) {
    double row_sum = 0.0;

    for (size_t j = 0; j < n; ++j) {
      if (i != j) {
        mat.At(i, j) = 1.0;
        row_sum += 1.0;
      }
    }

    mat.At(i, i) = row_sum * 5.0;

    b[i] = mat.At(i, i) + row_sum;
  }

  LinearSystem system(mat, b);
  system.epsilon = 1e-6;
  system.max_iterations = 10000;

  return system;
}

class ShakirovaESimpleIterationMethodPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  size_t matrix_size = 500;
  std::vector<double> expected_solution;

  void SetUp() override {
    matrix_size = 500;
    expected_solution.resize(matrix_size, 1.0);
  }

  InType GetTestInputData() override {
    return GenerateTestSystem(matrix_size);
  }

  bool CheckTestOutputData(OutType &output_data) override {
    auto run_type = std::get<2>(GetParam());
    auto task_name = std::get<1>(GetParam());

    if (run_type == ppc::performance::PerfResults::TypeOfRunning::kPipeline &&
        task_name.find("_seq") != std::string::npos) {
      return true;
    }

    if (ppc::util::IsUnderMpirun() && ppc::util::GetMPIRank() != 0) {
      return true;
    }

    if (output_data.size() != matrix_size) {
      return false;
    }

    for (size_t i = 0; i < matrix_size; ++i) {
      if (std::abs(output_data[i] - expected_solution[i]) > 1e-3) {
        return false;
      }
    }

    return true;
  }
};

TEST_P(ShakirovaESimpleIterationMethodPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShakirovaESimpleIterationMethodMPI, ShakirovaESimpleIterationMethodSEQ>(
        PPC_SETTINGS_shakirova_e_simple_iteration_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShakirovaESimpleIterationMethodPerfTest::CustomPerfTestName;
INSTANTIATE_TEST_SUITE_P(RunModeTests, ShakirovaESimpleIterationMethodPerfTest, kGtestValues, kPerfTestName);

}  // namespace shakirova_e_simple_iteration_method
