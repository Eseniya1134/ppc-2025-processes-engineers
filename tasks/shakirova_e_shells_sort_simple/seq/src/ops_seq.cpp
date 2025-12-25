#include "shakirova_e_shells_sort_simple/seq/include/ops_seq.hpp"

#include <vector>

#include "shakirova_e_shells_sort_simple/common/include/common.hpp"
#include "shakirova_e_shells_sort_simple/common/include/shell_sort.hpp"

namespace shakirova_e_shells_sort_simple {

ShakirovaEShellsSortSimpleSEQ::ShakirovaEShellsSortSimpleSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ShakirovaEShellsSortSimpleSEQ::ValidationImpl() {
  return true;
}

bool ShakirovaEShellsSortSimpleSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool ShakirovaEShellsSortSimpleSEQ::RunImpl() {
  if (GetOutput().empty()) {
    return true;
  }

  ShellSortImpl(GetOutput(), 0, static_cast<int>(GetOutput().size()) - 1);

  return true;
}

bool ShakirovaEShellsSortSimpleSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shakirova_e_shells_sort_simple
