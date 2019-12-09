// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

#include <vector>
#include <unordered_map>
#include <numeric>

#include <nbla/synced_array.hpp>
#include <nbla/computation_graph/function.hpp>

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
                  [&](const CgFunctionPtr &ptr) { scheduler.pre_function_callback(ptr); },
                  [&](const CgFunctionPtr &ptr) { scheduler.post_function_callback(ptr); });
    loss->variable()->grad()->fill(1.0);
    loss->backward(nullptr, true, {},
                   [&](const CgFunctionPtr &ptr) { scheduler.pre_function_callback(ptr); },
                   [&](const CgFunctionPtr &ptr) { scheduler.post_function_callback(ptr); });
    adam->update([&]() { swap_in_out_scheduler.pre_update_callback(); },
                 [&]() { swap_in_out_scheduler.post_update_callback(); });
    scheduler.end_scheduling();
    @endcode
 */
class SwapInOutScheduler {
  // Recorded tags of get/cast/clear
  // In the current implementation, get and cast were not distinguished.
  enum class RecTag {GETCAST, CLEAR};

  // Recorded information
  struct RecType {
    const RecTag tag;
    const unsigned int synced_array_id;
    weak_ptr<SyncedArray> sawptr;
    const Size_t size;
    const dtypes dtype;
    const Context ctx;
    bool preclear; // If true, the synced array can be cleared after this record.
    bool swapped_out; // If true, the synced arrat was swapped out.
    size_t swapped_out_bytes;
    bool no_need_swap_out; // If true, the synced arrat will not be swapped out.
  };

  const Context host_ctx; // Host context, the distination of swap out.
  const Context device_ctx; // Device context

  // The maximum size of usable GPU memory [byte]
  const size_t max_bytes_swap_in;  // for swap-in
  const size_t max_bytes_swap_out; // for waiting for the end of swap out

  // The used size of GPU memory [bytes]
  size_t used_bytes_swap_out = 0;

  // The recorded order of get/cast/clear in the first iteration
  vector<RecType> order;
  // In the rest iterations, the differently ordered get/cast/clear is recorded.
  vector<RecType> wrong_ordered;

  int order_idx = 0;   // pointing the current position in the order.
  int tail = 0;        // pointing the next record to wait for swap out
  size_t func_idx = 0; // pointing the current layer function.
  
  // Get/cast/clear used in each layer function were memorized as intervals
  // in the recorded order.
  vector<size_t> func_block_ends; 

  /* This is flags to monitor the occurence of preclear.
     Scheduled preclear can destroy data. It is safe as long as every iteration
     proceed in the same order as the recorded. However once disorder happens,
     scheduler breaks data in unexpected timing. If this dangerous situation
     is faced, the execution stops with error.
   */
  unordered_map<SyncedArrayPtr, bool> precleared;

  // Utility. This map is used only in the first iteration.
  unordered_map<SyncedArrayPtr, unsigned int> synced_array_id_mapper;

  // Utility. Map between SyncedArray ID and the order
  unordered_map<unsigned int, vector<int>> synced_array_id_to_order_idx;

  // Switch which separats the first iteration and others.
  bool first_iter = true;

public:
  /** Constructor.

  @param h_ctx Host context used as the destination of swap-out.
  @param d_ctx Device context.
  @param bytes Maximum GPU memory size managed by this class [bytes].
   */
  NBLA_API SwapInOutScheduler(const Context &h_ctx,
                              const Context &d_ctx,
                              const size_t bytes);

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
  void init();      // Called in start_scheduling()
  void finalize();  // Called in end_scheduling()

  // Common implementation of pre-function and pre-update callbacks
  void pre_callback();

  /* Pre callback function is separated the two parts.
     (1). The post-process of the previous function.
     (2). The pre-process of the next function.
     This is because NNabla calls "clear"s between post- and pre-callback
     functions and the scheduler should monitor these "clear"s.
  */
  void swap_out_step(); // (1)
  void swap_in_step();  // (2)

  synced_array_callback_func_type synced_array_callback;

  // SyncedArray callback function to record get/cast/clear in the first iteration.
  void synced_array_callback_recorder(SyncedArrayPtr saptr,
                                      const SyncedArrayCallbackTag func_name,
                                      const dtypes dtype,
                                      const Context &ctx,
                                      const bool write_only);

  // SyncedArray callback function to trace get/cast/clear after the first iteration.
  void synced_array_callback_tracer(SyncedArrayPtr saptr,
                                    const SyncedArrayCallbackTag func_name,
                                    const dtypes dtype,                                         
                                    const Context &ctx,
                                    const bool write_only);

  // Setter of appropriate SyncedArray callback function.
  void set_synced_array_callback();

  // Remover of SyncedArray callback function.
  void unset_synced_array_callback();

  void swap_in();
  void swap_out();
  
  // Subprocesses of swap_out()
  void swap_out_first_iter();
  void swap_out_scheduled();
  void wait_for_swap_out_first_iter();
  void wait_for_swap_out_scheduled();
  void wait_for_swap_out_first_iter_impl();
  void wait_for_swap_out_scheduled_impl(const RecType& r);

  // They used in finalization
  void wait_for_all_swap_out_first_iter();
  void wait_for_all_swap_out_scheduled();
  void swap_out_wrong_order();

  // Rename of long types to shorter
  using SyncedArrayCountsInQueue = unordered_map<unsigned int,
                                                 unordered_map<dtypes, int>>;
  using ScheduleType = vector<reference_wrapper<RecType>>;

  // Schedules
  unordered_map<int, ScheduleType> swap_in_schedule;
  unordered_map<int, ScheduleType> swap_out_schedule;
  unordered_map<int, ScheduleType> wait_schedule;
  ScheduleType wait_all_schedule;

  // Schedulers
  void schedule();
  void detect_swap_in_before_forward(int& head, size_t& used_bytes_swap_in,
                                     SyncedArrayCountsInQueue& synced_array_counts);
  ScheduleType
    schedule_swap_in(int& head, const int fid, size_t& used_bytes_swap_in,
                     SyncedArrayCountsInQueue& synced_array_counts,
                     unordered_map<unsigned int, bool>& host_uses_this_synced_array,
                     unordered_map<unsigned int, bool>& swapped_out,
                     unordered_map<unsigned int, RecType*>& swapped_out_r);
  ScheduleType
    schedule_swap_out(size_t& used_bytes_swap_in, 
                      SyncedArrayCountsInQueue& synced_array_counts,
                      const int fid,
                      unordered_map<unsigned int, bool>& swapped_out,
                      unordered_map<unsigned int, RecType*>& swapped_out_r);
  ScheduleType schedule_wait_for_swap_out(unordered_map<unsigned int, bool>& swapped_out,
                                          unordered_map<unsigned int, RecType*>& swapped_out_r);
  ScheduleType schedule_wait_for_all_swap_out(unordered_map<unsigned int, bool>& swapped_out,
                                              unordered_map<unsigned int, RecType*>& swapped_out_r);
  void schedule_wait_for_swap_out_impl(ScheduleType& schedule,
                                       unordered_map<unsigned int, bool>& swapped_out,
                                       unordered_map<unsigned int, RecType*>& swapped_out_r);
  void schedule_preclear();

  // Utility. Converter of tags
  RecTag get_tag(const SyncedArrayCallbackTag func_name, const bool write_only);
};
}
#endif
