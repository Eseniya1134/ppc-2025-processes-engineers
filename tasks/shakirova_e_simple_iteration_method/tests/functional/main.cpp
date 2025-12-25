#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "shakirova_e_simple_iteration_method/common/include/common.hpp"
#include "shakirova_e_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "shakirova_e_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace shakirova_e_simple_iteration_method {

class ShakirovaESimpleIterationMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    if (ShouldInitializeTestData()) {
      InitializeTestData();
      return;
    }

    SetEmptyData();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ShouldInitializeTestData()) {
      if (output_data.size() != expected_solution_.size()) {
        return false;
      }

      for (size_t i = 0; i < output_data.size(); i++) {
        if (std::abs(output_data[i] - expected_solution_[i]) > 1e-4) {
          return false;
        }
      }
      return true;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<double> expected_solution_;

  [[nodiscard]] static bool ShouldInitializeTestData() {
    const std::string &test_type =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());

    if (test_type.find("_mpi") == std::string::npos) {
      return true;
    }

    return !ppc::util::IsUnderMpirun() || ppc::util::GetMPIRank() == 0;
  }

  void SetEmptyData() {
    input_data_ = LinearSystem(0);
    expected_solution_.clear();
  }

  void InitializeTestData() {
    std::string test_name =
        std::get<1>(std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam())) + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_shakirova_e_simple_iteration_method, test_name);

    std::ifstream ifs(abs_path);

    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open test file: " + test_name);
    }

    size_t n = 0;
    size_t m = 0;
    ifs >> n >> m;

    if (n == 0 || m == 0 || n != m) {
      throw std::runtime_error("Both dimensions of matrix must be positive and equal");
    }

    Matrix mat(n, n);
    std::vector<double> b(n);
    expected_solution_.resize(n);

    std::string line;
    std::getline(ifs, line);

    for (size_t i = 0; i < n; i++) {
      std::getline(ifs, line);
      std::istringstream iss(line);

      for (size_t j = 0; j < n; j++) {
        double coeff = 0.0;
        char x_char = 0;
        char num_char = 0;

        if (j > 0) {
          char sign = 0;
          iss >> sign;
          iss >> coeff;
          if (sign == '-') {
            coeff = -coeff;
          }
        } else {
          iss >> coeff;
        }

        iss >> x_char >> num_char;
        mat.At(i, j) = coeff;
      }

      char equals = 0;
      iss >> equals;
      iss >> b[i];
    }

    std::getline(ifs, line);
    std::istringstream sol_stream(line);

    for (size_t i = 0; i < n; i++) {
      char x_char = 0;
      size_t idx = 0;
      char equals = 0;
      double value = 0.0;
      char comma = 0;

      sol_stream >> x_char >> idx >> equals >> value;
      expected_solution_[idx - 1] = value;

      if (i < n - 1) {
        sol_stream >> comma;
      }
    }

    input_data_ = LinearSystem(mat, b);
    input_data_.epsilon = 1e-6;
    input_data_.max_iterations = 1000;
  }
};

namespace {

TEST_P(ShakirovaESimpleIterationMethodFuncTests, SimpleIterationMethod) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {std::make_tuple(0, "test_1"), std::make_tuple(1, "test_2"),
                                            std::make_tuple(2, "test_3"), std::make_tuple(3, "test_4"),
                                            std::make_tuple(4, "test_5"), std::make_tuple(5, "test_6"),
                                            std::make_tuple(6, "test_7")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ShakirovaESimpleIterationMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_shakirova_e_simple_iteration_method),
                                           ppc::util::AddFuncTask<ShakirovaESimpleIterationMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_shakirova_e_simple_iteration_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    ShakirovaESimpleIterationMethodFuncTests::PrintFuncTestName<ShakirovaESimpleIterationMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(SimpleIterationTests, ShakirovaESimpleIterationMethodFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shakirova_e_simple_iteration_method
