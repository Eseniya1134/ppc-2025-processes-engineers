#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "shakirova_e_elem_matrix_sum/common/include/common.hpp"
#include "shakirova_e_elem_matrix_sum/common/include/matrix.hpp"
#include "shakirova_e_elem_matrix_sum/mpi/include/ops_mpi.hpp"
#include "shakirova_e_elem_matrix_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shakirova_e_elem_matrix_sum {

class ShakirovaEElemMatrixSumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<int64_t> input_elements;
    int64_t output_sum = 0;

    {
      std::string file_name =
          std::get<1>(std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam())) + ".txt";
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_shakirova_e_elem_matrix_sum, file_name);

      std::ifstream ifs(abs_path);

      if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open test file: " + file_name);
      }

      ifs >> rows >> cols;

      if (rows == 0 || cols == 0) {
        throw std::runtime_error("Both dimensions of matrix must be positive integers");
      }

      input_elements.resize(rows * cols);

      for (size_t i = 0; i < input_elements.size(); i++) {
        ifs >> input_elements[i];
      }

      ifs >> output_sum;
    }

    input_data_ = {.rows = rows, .cols = cols, .data = input_elements};
    output_data_ = output_sum;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = {};
  OutType output_data_ = 0;
};

namespace {

TEST_P(ShakirovaEElemMatrixSumFuncTests, MatrixElemSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {
    std::make_tuple(0, "test_1"), std::make_tuple(1, "test_2"), std::make_tuple(2, "test_3"),
    std::make_tuple(3, "test_4"), std::make_tuple(4, "test_5"), std::make_tuple(5, "test_6"),
    std::make_tuple(6, "test_7"), std::make_tuple(5, "test_8"), std::make_tuple(6, "test_9")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ShakirovaEElemMatrixSumMPI, InType>(kTestParam, PPC_SETTINGS_shakirova_e_elem_matrix_sum),
    ppc::util::AddFuncTask<ShakirovaEElemMatrixSumSEQ, InType>(kTestParam, PPC_SETTINGS_shakirova_e_elem_matrix_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShakirovaEElemMatrixSumFuncTests::PrintFuncTestName<ShakirovaEElemMatrixSumFuncTests>;

INSTANTIATE_TEST_SUITE_P(ElemSumTests, ShakirovaEElemMatrixSumFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shakirova_e_elem_matrix_sum
