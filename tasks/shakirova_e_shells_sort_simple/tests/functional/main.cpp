#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "shakirova_e_shells_sort_simple/common/include/common.hpp"
#include "shakirova_e_shells_sort_simple/common/include/shell_sort.hpp"
#include "shakirova_e_shells_sort_simple/mpi/include/ops_mpi.hpp"
#include "shakirova_e_shells_sort_simple/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shakirova_e_shells_sort_simple {

using TestParams = std::vector<int>;

class ShakirovaEShellsSortSimpleFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestParams> {
 public:
  static std::string PrintTestParam(const TestParams &p) {
    if (p.empty()) {
      return "EmptyVector";
    }
    std::string val = std::to_string(p[0]);
    if (p[0] < 0) {
      val = "Negative" + std::to_string(std::abs(p[0]));
    }
    return "Elements_" + std::to_string(p.size()) + "_StartVal_" + val;
  }

 protected:
  void SetUp() override {
    input_data_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank == 0) {
      std::vector<int> sorted_reference = input_data_;
      if (!sorted_reference.empty()) {
        ShellSortImpl(sorted_reference, 0, static_cast<int>(sorted_reference.size()) - 1);
      }
      return output_data == sorted_reference;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

std::vector<int> GenerateRandomVector(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-2000, 2000);
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

TEST_P(ShakirovaEShellsSortSimpleFuncTests, ShellSortValidation) {
  ExecuteTest(GetParam());
}

const std::array<TestParams, 11> kTestVectors = {TestParams{},
                                                 TestParams{42},
                                                 TestParams{9, 7, 5, 3, 1},
                                                 TestParams{10, 20, 30, 40, 50, 60},
                                                 TestParams{7, 7, 7, 7, 7},
                                                 TestParams{-8, -3, -6, -1, -9},
                                                 TestParams{-20, 15, -10, 25, 0, -5, 30},
                                                 TestParams{INT_MAX, 0, INT_MIN},
                                                 GenerateRandomVector(15),
                                                 GenerateRandomVector(75),
                                                 GenerateRandomVector(150)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ShakirovaEShellsSortSimpleMPI, InType>(
                                               kTestVectors, PPC_SETTINGS_shakirova_e_shells_sort_simple),
                                           ppc::util::AddFuncTask<ShakirovaEShellsSortSimpleSEQ, InType>(
                                               kTestVectors, PPC_SETTINGS_shakirova_e_shells_sort_simple));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = ShakirovaEShellsSortSimpleFuncTests::PrintFuncTestName<ShakirovaEShellsSortSimpleFuncTests>;

INSTANTIATE_TEST_SUITE_P(ShellSortFunctionalTests, ShakirovaEShellsSortSimpleFuncTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace shakirova_e_shells_sort_simple
