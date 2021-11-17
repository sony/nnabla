// Copyright 2021 Sony Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __NBLA_CUDA_PREFETCH_HPP__
#define __NBLA_CUDA_PREFETCH_HPP__

#include <numeric>
#include <set>
#include <unordered_map>
#include <vector>

#include <nbla/backend_registry.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/synced_array.hpp>

// Because older gcc cannot compile the hash of enum type,
// here is provided the definition of the hash of enum for dtypes.
// In this code, the hash of nbla::dtypes (enum) is used in std::unordered_map.
namespace std {
template <> struct hash<nbla::dtypes> {
  static_assert(is_enum<nbla::dtypes>::value,
                "This hash only works for enumeration types");
  size_t operator()(nbla::dtypes x) const noexcept {
    using type = typename underlying_type<nbla::dtypes>::type;
    return hash<type>{}(static_cast<type>(x));
  }
};
}

namespace nbla {

using std::vector;
using std::unordered_map;
using std::string;
using std::accumulate;
using std::weak_ptr;
using std::reference_wrapper;
using std::pair;
using std::set;

/** A class which manages GPU memory usage and schedules swap in/out
    throughout network computation.

    If GPU memory is insufficient to train your model,
    SwapInOutScheduler enables you to compute it fast and effectively
    by using memory swapping strategy.

    This class schedules the timing to swap out tensors to avoid out of memory
    and to swap in them before they are reused in computation.

    The schedule is made to be based on the usage order of tensors
    in the first training iteration. It indicates the disorder of
    the usage order in the rest iteration will cause speed-down.

    A scheduler takes the size of GPU memory which you want to manage.
    For example, when your GPU memory is up to 4 GB, the initialization is
    @code
    SwapInOutScheduler scheduler(cpu_ctx, gpu_ctx, 4e9);
    @endcode
    If out-of-memory error occurs in this configuration, the gradul reduction
    of 4e9 could solve the problem; for example, let the next size be 3.5e9.

    This scheduler can be used easily as extension by enclosing a training block
    between SwapInOutScheduler::start_scheduling() and
    SwapInOutScheduler::end_scheduling(). And also you need set the callback
    functions into the arguments of forward, backward, and update.
    For example, in a training loop,
    @code
    shceduler.start_scheduling();
    // Input next data and label in this line.
    loss->forward(false, true, nullptr,
                  [&](const CgFunctionPtr &ptr) {
          scheduler.pre_function_callback(ptr); },
                  [&](const CgFunctionPtr &ptr) {
          scheduler.post_function_callback(ptr); });
    loss->variable()->grad()->fill(1.0);
    loss->backward(nullptr, true, {},
                  [&](const CgFunctionPtr &ptr) {
          scheduler.pre_function_callback(ptr); },
                  [&](const CgFunctionPtr &ptr) {
          scheduler.post_function_callback(ptr); });
    adam->update([&]() { swap_in_out_scheduler.pre_update_callback(); },
                 [&]() { swap_in_out_scheduler.post_update_callback(); });
    scheduler.end_scheduling();
    @endcode
 */
class SwapInOutScheduler {
  // Notation: "sa" stands for SyncedArray

  //---------------------------------------------------
  //    Types
  //---------------------------------------------------
  // Recorded tags of get/cast/clear
  // In the current implementation, get and cast were not distinguished.
  enum class RecTag { GET, CAST, CLEAR };

  // Recorded information
  struct RecType {
    const RecTag tag;
    const unsigned int said;
    weak_ptr<SyncedArray> sawptr;
    const Size_t size;
    const dtypes dtype;
    const Context ctx;
    const bool write_only_cast;
    bool first_creation;
    bool temporary_buffer;

    RecType(const RecTag tag_, const unsigned int said_, SyncedArrayPtr saptr_,
            const Size_t size_, const dtypes dtype_, const Context ctx_,
            const bool write_only_cast_, const bool first_creation_)
        : tag(tag_), said(said_), sawptr(saptr_), size(size_), dtype(dtype_),
          ctx(ctx_), write_only_cast(write_only_cast_),
          first_creation(first_creation_), temporary_buffer(false) {}
  };

  //---------------------------------------------------
  //    Variables
  //---------------------------------------------------
  const Context host_ctx;   // Host context, the distination of swap out.
  const Context device_ctx; // Device context

  // The maximum size of usable GPU memory [byte]
  const size_t max_bytes;
  const size_t max_prefetch_bytes;

  // The recorded order of get/cast/clear in the first iteration
  vector<RecType> order;

  // The differently ordered get/cast/clear is recorded after first iteration
  vector<RecType> wrong_ordered;

  size_t order_idx = 0; // pointing the current position in the order.
  size_t func_idx = 0;  // pointing the current layer function.

  // Function blocks in the order
  vector<size_t> func_block_ends;

  // Flags to monitor preclear.
  unordered_map<SyncedArrayPtr, bool> precleared;

  // Flags to monitor cast prefetch.
  unordered_map<SyncedArrayPtr, bool> cast_prefetched;

  // Map: SyncedArray ID -> the indices in order
  unordered_map<unsigned int, vector<int>> said_to_order_idx;

  // Switch the first iteration and others.
  bool first_iter = true;
  bool second_iter = false;

  // Check whether function blocks have get/cast on host
  vector<bool> is_host_func;
  void determine_which_is_host_func();

  // This option controlls whether prefeth uses cast().
  const bool cast_prefetch;
  const bool cast_prefetch_no_abort;
  const bool free_host_caches;

  //---------------------------------------------------
  //    Variables used only in first iteration
  //---------------------------------------------------
  // Map: SyncedArrayPtr -> SyncedArray ID
  unordered_map<SyncedArrayPtr, unsigned int> said_map;

public:
  //---------------------------------------------------
  //               User interfaces
  //---------------------------------------------------
  /** Constructor.

  @params h_ctx Host context used as the destination of swap-out.
  @params d_ctx Device context.
  @params max Maximum GPU memory size managed by this class [bytes].
  @params prefetch_max Maximum prefetch length.
  @params save_host_mem The flag to switch prefetch scheme to save host memory.
  @params save_host_mem_no_abort Irregular off-schedule does not abort program
                                 if cast prefetch irreversibly change the type
                                 of array.
  */
  NBLA_API SwapInOutScheduler(const Context &h_ctx, const Context &d_ctx,
                              const size_t max, const size_t prefetch_max = 0,
                              const bool save_host_mem = true,
                              const bool save_host_mem_no_abort = false);

  /** Destructor.
   */
  NBLA_API ~SwapInOutScheduler();

  /** This initializes the scheduler and starts the management of GPU memory
      in this iteration.
   */
  NBLA_API void start_scheduling();

  /** This finalizes the scheduler and stops the management of GPU memory
      in this iteration.
   */
  NBLA_API void end_scheduling();

  /** To use the scheduler, this callback must be set in the pre-function-hook
      arguments of forward and backward function.
   */
  NBLA_API void pre_function_callback(const CgFunctionPtr &ptr);

  /** To use the scheduler, this callback must be set in the post-function-hook
      arguments of forward and backward function.
   */
  NBLA_API void post_function_callback(const CgFunctionPtr &ptr);

  /** To use the scheduler, this callback must be set in the pre-update-hook
      argument of the update method of a solver.
   */
  NBLA_API void pre_update_callback();

  /** To use the scheduler, this callback must be set in the post-update-hook
      argument of the update method of a solver.
   */
  NBLA_API void post_update_callback();

private:
  //---------------------------------------------------
  //                   Scheduler
  //---------------------------------------------------
  enum class ScheduleTag {
    SWAP_IN_GET,
    SWAP_IN_CAST,
    SWAP_OUT,
    WAIT,
    PRECLEAR
  };

  struct ScheduleType {
    ScheduleTag tag;
    RecType *r;

    ScheduleType(const ScheduleTag tag_, RecType *r_) : tag(tag_), r(r_) {}

    ScheduleType &operator=(const ScheduleType &other) {
      tag = other.tag;
      r = other.r;
      return *this;
    }
  };

  enum class ArrayStateTag {
    CLEARED,
    IN,
    OUT,
    UNPREFETCHED,
    OUT_WAITED,
    OUT_CLEARED
  };

  struct ArrayState {
    int count = 0;
    ArrayStateTag state = ArrayStateTag::CLEARED;
    RecType *swapped_out_r = nullptr;
  };

  // Rename of long types to shorter
  using SyncedArrayStates =
      unordered_map<unsigned int, unordered_map<dtypes, ArrayState>>;
  struct ScheduleParams {
    int head = 0;
    int tail = 0;
    int fid = 0;
    size_t prefetch_bytes = 0;
    size_t swap_in_bytes = 0;
    size_t swap_out_bytes = 0;
    SyncedArrayStates sa_states;
  };

  // Schedules
  vector<vector<ScheduleType>> beginning_schedules;
  vector<vector<ScheduleType>> end_schedules;

  // Main function
  void schedule();

  // Subprocesses of shcedule()
  void calc_mem_usage_before_forward(
      ScheduleParams &params,
      unordered_map<unsigned int, pair<bool, dtypes>> &head_type);

  void
  schedule_swap_in(ScheduleParams &params,
                   const vector<unsigned int> prefetch_stopper,
                   unordered_map<unsigned int, bool> &type_converted,
                   unordered_map<unsigned int, pair<bool, dtypes>> &head_type);

  void schedule_swap_out(
      ScheduleParams &params,
      unordered_map<unsigned int, vector<pair<RecType *, bool>>> &clear_info,
      unordered_map<unsigned int, pair<bool, dtypes>> &head_type);

  void schedule_wait_for_all_swap_out(ScheduleParams &params);

  void schedule_wait_for_swap_out_impl(ScheduleParams &params);

  void schedule_preclear(
      unordered_map<unsigned int, vector<pair<RecType *, bool>>> &clear_info);

  // Return a flag to decide whether to do reschedule.
  bool reserve_unprefetched_memory(ScheduleParams &params,
                                   vector<unsigned int> &prefetch_stopper);

  void reconfirm_first_creation();

  void collect_info_about_dtype_conversion(
      unordered_map<unsigned int, bool> &type_converted,
      const unordered_map<unsigned int, pair<bool, dtypes>> &head_type);

  // Return true when a recorded get/cast transfer data between host and device.
  bool no_data_transfer(const RecType *r);

  void cancel_swap_out(const RecType *r, ScheduleParams &params);

  void backtrack_with_prefetch_cancel(ScheduleParams &params,
                                      vector<unsigned int> &prefetch_stopper,
                                      const size_t unprefetched_bytes,
                                      size_t available_bytes);

  void determine_first_head_types(
      unordered_map<unsigned int, pair<bool, dtypes>> &head_dtype);

  // Return true when memory is not enough to prefetch the next array.
  bool free_memory_to_prefetch(ScheduleParams &params,
                               const size_t next_array_bytes);

  int accumulate_counts(const unordered_map<dtypes, ArrayState> &count_map);

  // Check if an array is not cleared. When a SyncedArray is empty,
  // the scheduler should not call get/cast not to create a unnecessary array.
  inline bool is_not_cleared_yet(const SyncedArrayPtr saptr) {
    return saptr->get_num_arrays() > 0;
  }

  bool context_checker(const Context query_ctx, const Context ctx) {
    auto array_classes = BackendUtils::array_classes(ctx);

    return std::find(array_classes.begin(), array_classes.end(),
                     query_ctx.array_class) != array_classes.end();
  }

  //---------------------------------------------------
  //               Swap in/out
  //---------------------------------------------------
  // Common implementation of function and update callbacks
  void pre_callback();
  void post_callback();

  // Subprocesses of swap_out()
  void swap_out_first_iter();

  // Swap out disordered arrays in finalization
  void swap_out_wrong_order();

  // Execute swap in/out, wait, and preclear on a schedule.
  void run_on_beginning_schedule();
  void run_on_end_schedule();
  void run(const ScheduleType &s);

  //---------------------------------------------------
  //              SyncedArrayCallback
  //---------------------------------------------------
  synced_array_callback_func_type sa_callback;

  // Setter
  void set_sa_callback();

  // Unsetter
  void unset_sa_callback();

  // SyncedArrayCallback to record get/cast/clear in the first iteration.
  void sa_callback_recorder(SyncedArrayPtr saptr,
                            const SyncedArrayCallbackTag sa_tag,
                            const dtypes dtype, const Context &ctx,
                            const bool write_only, const bool first_creation,
                            const bool off_recording);

  // SyncedArrayCallback to trace get/cast/clear after the first iteration.
  void sa_callback_tracer(SyncedArrayPtr saptr,
                          const SyncedArrayCallbackTag sa_tag,
                          const dtypes dtype, const Context &ctx,
                          const bool write_only, const bool first_creation,
                          const bool off_recording);

  // Tag converter
  RecTag convert_tag(const SyncedArrayCallbackTag sa_tag,
                     const bool write_only);

  string to_str(const ScheduleTag &);
};
}
#endif
