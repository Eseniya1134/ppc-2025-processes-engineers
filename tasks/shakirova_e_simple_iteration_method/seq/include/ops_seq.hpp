#pragma once

#include "shakirova_e_simple_iteration_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shakirova_e_simple_iteration_method {

class ShakirovaESimpleIterationMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShakirovaESimpleIterationMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shakirova_e_simple_iteration_method
