#pragma once

#include <string>
#include <tuple>

#include "linear_system.hpp"
#include "task/include/task.hpp"

namespace shakirova_e_simple_iteration_method {

using InType = LinearSystem;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shakirova_e_simple_iteration_method
