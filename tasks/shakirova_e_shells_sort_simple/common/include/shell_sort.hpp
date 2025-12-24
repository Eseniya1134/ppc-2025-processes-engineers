#pragma once
#include <vector>

namespace shakirova_e_shells_sort_simple {

inline void ShellSortImpl(std::vector<int> &vec, int left, int right) {
  int n = right - left + 1;
  if (n <= 1) {
    return;
  }

  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = left + gap; i <= right; i++) {
      int temp = vec[i];
      int j = i;

      while (j >= left + gap && vec[j - gap] > temp) {
        vec[j] = vec[j - gap];
        j -= gap;
      }
      vec[j] = temp;
    }
  }
}

}  // namespace shakirova_e_shells_sort_simple
