#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>

#include "shakirova_e_simple_iteration_method/common/include/common.hpp"
#include "shakirova_e_simple_iteration_method/mpi/include/ops_mpi.hpp"
#define private public
#include "shakirova_e_simple_iteration_method/seq/include/ops_seq.hpp"
#undef private
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
      throw std::runtime_error("Invalid system dimensions");
    }

    Matrix A(n, n);
    std::vector<double> b(n);
    expected_solution_.resize(n);

    std::string line;
    std::getline(ifs, line);
    
    for (size_t i = 0; i < n; i++) {
      std::getline(ifs, line);
      std::istringstream iss(line);
      
      for (size_t j = 0; j < n; j++) {
        double coeff;
        char x_char, num_char;
        
        if (j > 0) {
          char sign;
          iss >> sign;
          iss >> coeff;
          if (sign == '-') coeff = -coeff;
        } else {
          iss >> coeff;
        }
        
        iss >> x_char >> num_char;
        A.At(i, j) = coeff;
      }
      
      char equals;
      iss >> equals;
      iss >> b[i];
    }

    std::getline(ifs, line);
    std::istringstream sol_stream(line);
    
    for (size_t i = 0; i < n; i++) {
      char x_char;
      size_t idx;
      char equals;
      double value;
      char comma;
      
      sol_stream >> x_char >> idx >> equals >> value;
      expected_solution_[idx - 1] = value;
      
      if (i < n - 1) {
        sol_stream >> comma;
      }
    }

    input_data_ = LinearSystem(A, b);
    input_data_.epsilon = 1e-6;
    input_data_.max_iterations = 1000;
  }
};

namespace {

TEST_P(ShakirovaESimpleIterationMethodFuncTests, SimpleIterationMethod) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {
    std::make_tuple(0, "test_1"), std::make_tuple(1, "test_2"), std::make_tuple(2, "test_3"),
    std::make_tuple(3, "test_4"), std::make_tuple(4, "test_5"), std::make_tuple(5, "test_6"),
    std::make_tuple(6, "test_7"), std::make_tuple(7, "test_8"), std::make_tuple(8, "test_9"),
    std::make_tuple(9, "test_10"), std::make_tuple(10, "test_11"), std::make_tuple(11, "test_12")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ShakirovaESimpleIterationMethodMPI, InType>(kTestParam, PPC_SETTINGS_shakirova_e_simple_iteration_method),
    ppc::util::AddFuncTask<ShakirovaESimpleIterationMethodSEQ, InType>(kTestParam, PPC_SETTINGS_shakirova_e_simple_iteration_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShakirovaESimpleIterationMethodFuncTests::PrintFuncTestName<ShakirovaESimpleIterationMethodFuncTests>;

// ВАЖНО: Измените имя test suite
INSTANTIATE_TEST_SUITE_P(SimpleIterationTests, ShakirovaESimpleIterationMethodFuncTests, kGtestValues, kPerfTestName);

}  // namespace

// Unit тесты с правильным префиксом
TEST(ShakirovaESimpleIterationMethod_UnitTests, InvalidSystemValidation) {
  LinearSystem system(0);
  ShakirovaESimpleIterationMethodSEQ task(system);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(ShakirovaESimpleIterationMethod_UnitTests, ZeroDiagonalValidation) {
  Matrix A(2, 2);
  A.At(0, 0) = 0.0;  A.At(0, 1) = 1.0;
  A.At(1, 0) = 1.0;  A.At(1, 1) = 4.0;
  
  std::vector<double> b = {5.0, 5.0};
  LinearSystem system(A, b);
  
  ShakirovaESimpleIterationMethodSEQ task(system);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(ShakirovaESimpleIterationMethod_UnitTests, NoDiagonalDominanceValidation) {
  Matrix A(2, 2);
  A.At(0, 0) = 1.0;  A.At(0, 1) = 5.0;
  A.At(1, 0) = 5.0;  A.At(1, 1) = 1.0;
  
  std::vector<double> b = {6.0, 6.0};
  LinearSystem system(A, b);
  
  ShakirovaESimpleIterationMethodSEQ task(system);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(ShakirovaESimpleIterationMethod_UnitTests, ValidSystemWith3x3) {
  Matrix A(3, 3);
  A.At(0, 0) = 10.0;  A.At(0, 1) = 1.0;   A.At(0, 2) = 1.0;
  A.At(1, 0) = 1.0;   A.At(1, 1) = 10.0;  A.At(1, 2) = 1.0;
  A.At(2, 0) = 1.0;   A.At(2, 1) = 1.0;   A.At(2, 2) = 10.0;
  
  std::vector<double> b = {12.0, 12.0, 12.0};
  LinearSystem system(A, b);
  
  ShakirovaESimpleIterationMethodSEQ task(system);
  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
  
  auto solution = task.GetOutput();
  EXPECT_EQ(solution.size(), 3);
  EXPECT_NEAR(solution[0], 1.0, 1e-5);
  EXPECT_NEAR(solution[1], 1.0, 1e-5);
  EXPECT_NEAR(solution[2], 1.0, 1e-5);
}

TEST(ShakirovaESimpleIterationMethod_UnitTests, MatrixNormCalculation) {
  Matrix A(2, 2);
  A.At(0, 0) = 5.0;  A.At(0, 1) = 1.0;
  A.At(1, 0) = 1.0;  A.At(1, 1) = 4.0;
  
  std::vector<double> b = {6.0, 5.0};
  LinearSystem system(A, b);
  
  double norm = system.MatrixNorm(A);
  EXPECT_NEAR(norm, 6.0, 1e-10);
}

TEST(ShakirovaESimpleIterationMethod_UnitTests, VectorNormCalculation) {
  std::vector<double> v = {3.0, -4.0, 1.0};
  double norm = LinearSystem::VectorNorm(v);
  EXPECT_NEAR(norm, 4.0, 1e-10);
}

TEST(ShakirovaESimpleIterationMethod_UnitTests, TransformToIterationForm) {
  Matrix A(2, 2);
  A.At(0, 0) = 5.0;  A.At(0, 1) = 1.0;
  A.At(1, 0) = 1.0;  A.At(1, 1) = 4.0;
  
  std::vector<double> b = {6.0, 5.0};
  LinearSystem system(A, b);
  
  Matrix B;
  std::vector<double> c;
  EXPECT_TRUE(system.TransformToIterationForm(B, c));
  
  EXPECT_NEAR(c[0], 1.2, 1e-10);
  EXPECT_NEAR(c[1], 1.25, 1e-10);
  EXPECT_NEAR(B.At(0, 1), -0.2, 1e-10);
  EXPECT_NEAR(B.At(1, 0), -0.25, 1e-10);
}

TEST(ShakirovaESimpleIterationMethod_UnitTests, SetInitialGuess) {
  Matrix A(2, 2);
  A.At(0, 0) = 5.0;  A.At(0, 1) = 1.0;
  A.At(1, 0) = 1.0;  A.At(1, 1) = 4.0;
  
  std::vector<double> b = {6.0, 5.0};
  LinearSystem system(A, b);
  
  std::vector<double> initial_guess = {0.5, 0.5};
  system.SetInitialGuess(initial_guess);
  
  EXPECT_EQ(system.x[0], 0.5);
  EXPECT_EQ(system.x[1], 0.5);
}

TEST(ShakirovaESimpleIterationMethod_UnitTests, InvalidInitialGuessSize) {
  Matrix A(2, 2);
  A.At(0, 0) = 5.0;  A.At(0, 1) = 1.0;
  A.At(1, 0) = 1.0;  A.At(1, 1) = 4.0;
  
  std::vector<double> b = {6.0, 5.0};
  LinearSystem system(A, b);
  
  std::vector<double> wrong_guess = {0.5, 0.5, 0.5};
  EXPECT_THROW(system.SetInitialGuess(wrong_guess), std::invalid_argument);
}

TEST(ShakirovaESimpleIterationMethod_UnitTests, DiagonalDominanceCheck) {
  Matrix A(3, 3);
  A.At(0, 0) = 10.0;  A.At(0, 1) = 1.0;   A.At(0, 2) = 1.0;
  A.At(1, 0) = 1.0;   A.At(1, 1) = 10.0;  A.At(1, 2) = 1.0;
  A.At(2, 0) = 1.0;   A.At(2, 1) = 1.0;   A.At(2, 2) = 10.0;
  
  std::vector<double> b = {12.0, 12.0, 12.0};
  LinearSystem system(A, b);
  
  EXPECT_TRUE(system.HasDiagonalDominance());
}

TEST(ShakirovaESimpleIterationMethod_UnitTests, ConvergenceWithTightTolerance) {
  Matrix A(2, 2);
  A.At(0, 0) = 5.0;  A.At(0, 1) = 1.0;
  A.At(1, 0) = 1.0;  A.At(1, 1) = 4.0;
  
  std::vector<double> b = {6.0, 5.0};
  LinearSystem system(A, b);
  system.epsilon = 1e-10;
  
  ShakirovaESimpleIterationMethodSEQ task(system);
  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
}

TEST(ShakirovaESimpleIterationMethod_UnitTests, MatrixEqualityCheck) {
  Matrix A1(2, 2);
  Matrix A2(2, 2);
  
  A1.At(0, 0) = 1.0;  A1.At(0, 1) = 2.0;
  A1.At(1, 0) = 3.0;  A1.At(1, 1) = 4.0;
  
  A2.At(0, 0) = 1.0;  A2.At(0, 1) = 2.0;
  A2.At(1, 0) = 3.0;  A2.At(1, 1) = 4.0;
  
  EXPECT_TRUE(A1 == A2);
}

}  // namespace shakirova_e_simple_iteration_method