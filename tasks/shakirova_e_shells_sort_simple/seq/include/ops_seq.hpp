#pragma once

#include "shakirova_e_shells_sort_simple/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shakirova_e_shells_sort_simple {

class ShakirovaEShellsSortSimpleSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShakirovaEShellsSortSimpleSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shakirova_e_shells_sort_simple
