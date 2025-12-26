#pragma once
#include <cstddef>
#include <vector>

#include "shakirova_e_simple_iteration_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shakirova_e_simple_iteration_method {

class ShakirovaESimpleIterationMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShakirovaESimpleIterationMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SyncConfiguration(int rank, size_t &dimension, double &tolerance, size_t &max_iter);

  bool PrepareSystemMatrices(int rank, size_t dimension, std::vector<double> &b_flat, std::vector<double> &c_vector,
                             std::vector<double> &x_current);

  static bool CheckConvergence(int rank, size_t dimension, double tolerance, const std::vector<double> &x_next,
                               std::vector<double> &x_current);
};

}  // namespace shakirova_e_simple_iteration_method
