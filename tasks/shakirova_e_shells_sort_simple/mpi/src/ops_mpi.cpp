#include "shakirova_e_shells_sort_simple/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "shakirova_e_shells_sort_simple/common/include/common.hpp"
#include "shakirova_e_shells_sort_simple/common/include/shell_sort.hpp"

namespace shakirova_e_shells_sort_simple {

ShakirovaEShellsSortSimpleMPI::ShakirovaEShellsSortSimpleMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ShakirovaEShellsSortSimpleMPI::ValidationImpl() {
  return true;
}

bool ShakirovaEShellsSortSimpleMPI::PreProcessingImpl() {
  return true;
}

bool ShakirovaEShellsSortSimpleMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_elements = 0;
  if (rank == 0) {
    total_elements = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_elements == 0) {
    return true;
  }

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  int base_count = total_elements / size;
  int remainder = total_elements % size;
  int current_displ = 0;

  for (int i = 0; i < size; ++i) {
    send_counts[i] = base_count + (i < remainder ? 1 : 0);
    displs[i] = current_displ;
    current_displ += send_counts[i];
  }

  std::vector<int> local_vec(send_counts[rank]);
  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, send_counts.data(), displs.data(), MPI_INT, local_vec.data(),
               send_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  if (!local_vec.empty()) {
    ShellSortImpl(local_vec, 0, static_cast<int>(local_vec.size()) - 1);
  }

  if (rank == 0) {
    GetOutput().resize(total_elements);
  }

  MPI_Gatherv(local_vec.data(), send_counts[rank], MPI_INT, rank == 0 ? GetOutput().data() : nullptr,
              send_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    auto current_end = GetOutput().begin() + send_counts[0];

    for (int i = 1; i < size; ++i) {
      if (send_counts[i] > 0) {
        auto next_end = current_end + send_counts[i];
        std::inplace_merge(GetOutput().begin(), current_end, next_end);
        current_end = next_end;
      }
    }
  }

  return true;
}

bool ShakirovaEShellsSortSimpleMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shakirova_e_shells_sort_simple
