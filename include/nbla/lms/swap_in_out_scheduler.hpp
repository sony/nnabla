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
  // Notation: "sa" stands for SyncedArray

  //---------------------------------------------------
  //    Types
  //---------------------------------------------------
  // Recorded tags of get/cast/clear
  // In the current implementation, get and cast were not distinguished.
  enum class RecTag {GETCAST, CLEAR};

  // Recorded information
  struct RecType {
    const RecTag tag;
    const unsigned int said;
    weak_ptr<SyncedArray> sawptr;
    const Size_t size;
    const dtypes dtype;
    const Context ctx;
    const bool in_func;
    const bool no_data_transfer;

    bool swapped_out = false; // If true, the synced array was swapped out.
    size_t swapped_out_bytes = 0;

    RecType(const RecTag tag_, const unsigned int said_, SyncedArrayPtr saptr_,
            const Size_t size_, const dtypes dtype_, const Context ctx_,
            const bool in_func_, const bool no_data_transfer_)
    : tag(tag_), said(said_), sawptr(saptr_), 
      size(size_), dtype(dtype_), ctx(ctx_), in_func(in_func_),
      no_data_transfer(no_data_transfer_) {}
  };


  //---------------------------------------------------
  //    Variables
  //---------------------------------------------------
  const Context host_ctx; // Host context, the distination of swap out.
  const Context device_ctx; // Device context

  // The maximum size of usable GPU memory [byte]
  const size_t max_bytes;
  const size_t max_prefetch_bytes;

  // The recorded order of get/cast/clear in the first iteration
  vector<RecType> order;
  
  // The differently ordered get/cast/clear is recorded after first iteration
  vector<RecType> wrong_ordered;

  int order_idx = 0;   // pointing the current position in the order.
  size_t func_idx = 0; // pointing the current layer function.

  // Function blocks in the order
  vector<size_t> func_block_ends;

  // Flags to monitor preclear.
  unordered_map<SyncedArrayPtr, bool> precleared;

  // Map: SyncedArray ID -> the indices in order
  unordered_map<unsigned int, vector<int>> said_to_order_idx;

  // Switch the first iteration and others.
  bool first_iter = true;

  // Check whether function blocks have get/cast on host
  vector<bool> is_host_func;
  void check_which_is_host_func();


  //---------------------------------------------------
  //    Variables used only in first iteration
  //---------------------------------------------------
  // The used size of GPU memory [byte]
  size_t used_bytes_swap_out_first_iter = 0;

  int tail_first_iter = 0; // pointing the next record to wait for swap out
    
  // Map: SyncedArrayPtr -> SyncedArray ID
  unordered_map<SyncedArrayPtr, unsigned int> said_map;



public:
  //---------------------------------------------------
  //               User interfaces
  //---------------------------------------------------
  /** Constructor.

  @param h_ctx Host context used as the destination of swap-out.
  @param d_ctx Device context.
  @param bytes Maximum GPU memory size managed by this class [bytes].
  @param prefetch_bytes Maximum prefetch length.
   */
  NBLA_API SwapInOutScheduler(const Context &h_ctx,
                              const Context &d_ctx,
                              const size_t bytes,
                              const size_t prefetch_bytes);

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
  enum class ScheduleTag { SWAP_IN, SWAP_OUT, WAIT };

  struct ScheduleType {
    ScheduleTag tag;
    RecType *r;

    ScheduleType(const ScheduleTag tag_, RecType *r_)
      : tag(tag_), r(r_) {}

    ScheduleType& operator=(const ScheduleType& other) {
      tag = other.tag;
      r = other.r;
      return *this;
    }
  };

  // Execute swap in/out, wait, and preclear on a schedule.
  void run_on_schedule();

  // Rename of long types to shorter
  using SyncedArrayCounts = unordered_map<unsigned int, unordered_map<dtypes, int>>;

  // Schedules
  unordered_map<int, vector<ScheduleType>> schedules_swap;
  unordered_map<int, vector<RecType*>> preclear_schedule;

  // Main function
  void schedule();

  // Subprocesses of shcedule()
  void calc_mem_usage_before_forward(int& head, size_t& prefetch_bytes,
                                     size_t& used_bytes_swap_in,
                                     SyncedArrayCounts& synced_array_counts);
  
  void schedule_swap_in
   (const bool pre, int& head, int& tail, const int fid, size_t& prefetch_bytes,
    size_t& used_bytes_swap_in, size_t& used_bytes_swap_out,
    SyncedArrayCounts& synced_array_counts,
    unordered_map<unsigned int, bool>& host_uses_this_synced_array,
    unordered_map<unsigned int, unordered_map<dtypes, bool>>& swapped_out,
    unordered_map<unsigned int, RecType*>& swapped_out_r,
    vector<RecType*>& canceled_swap_out, 
    vector<bool>& unprefetched);
  
  void schedule_swap_out
   (const int fid, size_t& prefetch_bytes, 
    size_t& used_bytes_swap_in, size_t& used_bytes_swap_out,
    SyncedArrayCounts& synced_array_counts,
    unordered_map<unsigned int, unordered_map<dtypes, bool>>& swapped_out,
    unordered_map<unsigned int, RecType*>& swapped_out_r);
  
  void schedule_wait_for_all_swap_out
   (const int fid, int& tail, size_t& used_bytes_swap_out,
    unordered_map<unsigned int, unordered_map<dtypes, bool>>& swapped_out,
    unordered_map<unsigned int, RecType*>& swapped_out_r,
    vector<RecType*>& canceled_swap_out);
  
  void schedule_wait_for_swap_out_impl
   (const int fid, int& tail, size_t& used_bytes_swap_out,
    unordered_map<unsigned int, unordered_map<dtypes, bool>>& swapped_out,
    unordered_map<unsigned int, RecType*>& swapped_out_r,
    vector<RecType*>& canceled_swap_out);
  
  void schedule_preclear();

  void cancel_swap_out(vector<RecType*>& canceled_swap_out);

  void reserve_unprefetched_memory(int& tail, const int fid, 
                                   size_t& prefetch_bytes,
                                   size_t& used_bytes_swap_in,
                                   size_t& used_bytes_swap_out,
                                   unordered_map<unsigned int, unordered_map<dtypes, bool>>& swapped_out,
                                   unordered_map<unsigned int, RecType*>& swapped_out_r,
                                   vector<RecType*>& canceled_swap_out,
                                   vector<bool>& unprefetched);


  //---------------------------------------------------
  //               Swap in/out
  //---------------------------------------------------
  // Common implementation of function and update callbacks
  void pre_callback();
  void post_callback();

  /* Pre callback function is separated the two parts.
  (1). The post-process of the previous function.
  (2). The pre-process of the next function.
  This is because NNabla calls "clear"s between post- and pre-callback
  functions and the scheduler should monitor these "clear"s.
  */

  // Subprocesses of swap_out()
  void swap_out_first_iter();
  void wait_for_swap_out_first_iter();
  void wait_for_all_swap_out_first_iter();
  void wait_for_swap_out_first_iter_impl();

  // Swap out disordered arrays in finalization
  void swap_out_wrong_order();

  // Check if an array is not cleared. When a SyncedArray is empty,
  // the scheduler should not call get/cast not to create a unnecessary array.
  bool is_not_cleared_yet(const SyncedArrayPtr saptr) {
    return saptr->get_num_arrays() > 0;
  }

  //---------------------------------------------------
  //              SyncedArrayCallback
  //---------------------------------------------------
  synced_array_callback_func_type synced_array_callback;

  // Setter
  void set_synced_array_callback();

  // Unsetter
  void unset_synced_array_callback();

  // SyncedArrayCallback to record get/cast/clear in the first iteration.
  void synced_array_callback_recorder(SyncedArrayPtr saptr,
                                      const SyncedArrayCallbackTag sa_tag,
                                      const dtypes dtype,
                                      const Context &ctx,
                                      const bool write_only,
                                      const bool first_creation);

  // SyncedArrayCallback to trace get/cast/clear after the first iteration.
  void synced_array_callback_tracer(SyncedArrayPtr saptr,
                                    const SyncedArrayCallbackTag sa_tag,
                                    const dtypes dtype,
                                    const Context &ctx,
                                    const bool write_only,
                                    const bool first_creation);

  // Tag converter
  RecTag convert_tag(const SyncedArrayCallbackTag sa_tag,
                     const bool write_only);
};
}
#endif
