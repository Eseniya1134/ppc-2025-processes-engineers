#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <random>
#include <sstream>
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

static int test_counter = 0;

class ShakirovaEShellsSortSimpleFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestParams> {
 public:
  static std::string PrintTestParam(const TestParams &p) {
    std::string unique_id = "Test_" + std::to_string(test_counter++);

    if (p.empty()) {
      return unique_id + "_EmptyVector";
    }

    std::string val = std::to_string(p[0]);
    if (p[0] < 0) {
      val = "Neg" + std::to_string(std::abs(p[0]));
    }

    return unique_id + "_Size_" + std::to_string(p.size()) + "_First_" + val;
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

std::vector<int> GenerateRandomVector(size_t size, int seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(-2000, 2000);
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

std::vector<int> ReadVectorFromFile(const std::string &filename) {
  std::vector<int> result;
  std::ifstream file(filename);

  if (!file.is_open()) {
    return {-999999};
  }

  std::string line;
  if (std::getline(file, line)) {
    std::istringstream iss(line);
    int value = 0;
    while (iss >> value) {
      result.push_back(value);
    }
  }

  file.close();

  if (result.empty()) {
    return {-999999};
  }

  return result;
}

TEST_P(ShakirovaEShellsSortSimpleFuncTests, ShellSortValidation) {
  ExecuteTest(GetParam());
}

// Встроенные тесты
const std::array<TestParams, 11> kTestVectors = {TestParams{},
                                                 TestParams{42},
                                                 TestParams{9, 7, 5, 3, 1},
                                                 TestParams{10, 20, 30, 40, 50, 60},
                                                 TestParams{7, 7, 7, 7, 7},
                                                 TestParams{-8, -3, -6, -1, -9},
                                                 TestParams{-20, 15, -10, 25, 0, -5, 30},
                                                 TestParams{INT_MAX, 0, INT_MIN},
                                                 GenerateRandomVector(15, 100),
                                                 GenerateRandomVector(75, 200),
                                                 GenerateRandomVector(150, 300)};

// Тесты из файлов
const std::array<TestParams, 4> kFileTestVectors = {
    ReadVectorFromFile("data/test_1.txt"), ReadVectorFromFile("data/test_2.txt"), ReadVectorFromFile("data/test_3.txt"),
    ReadVectorFromFile("data/test_4.txt")};

// Создаем задачи
const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ShakirovaEShellsSortSimpleMPI, InType>(
                                               kTestVectors, PPC_SETTINGS_shakirova_e_shells_sort_simple),
                                           ppc::util::AddFuncTask<ShakirovaEShellsSortSimpleSEQ, InType>(
                                               kTestVectors, PPC_SETTINGS_shakirova_e_shells_sort_simple),
                                           ppc::util::AddFuncTask<ShakirovaEShellsSortSimpleMPI, InType>(
                                               kFileTestVectors, PPC_SETTINGS_shakirova_e_shells_sort_simple),
                                           ppc::util::AddFuncTask<ShakirovaEShellsSortSimpleSEQ, InType>(
                                               kFileTestVectors, PPC_SETTINGS_shakirova_e_shells_sort_simple));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = ShakirovaEShellsSortSimpleFuncTests::PrintFuncTestName<ShakirovaEShellsSortSimpleFuncTests>;

INSTANTIATE_TEST_SUITE_P(ShellSortFunctionalTests, ShakirovaEShellsSortSimpleFuncTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace shakirova_e_shells_sort_simple
