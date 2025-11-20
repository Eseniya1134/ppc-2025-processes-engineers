#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shakirova_e_elem_matrix_sum {

using InType = std::vector<std::vector<int>>;
using OutType = long long;
using TestType = std::tuple<int, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shakirova_e_elem_matrix_sum
