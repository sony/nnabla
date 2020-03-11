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


#include <limits>
#include <numeric>

#include <nbla/lms/swap_in_out_scheduler.hpp>
#include <nbla/singleton_manager.hpp>


#define SWAPINOUTSCHEDULER_DEBUG true


namespace nbla {

using std::accumulate;

// Constructor
SwapInOutScheduler::SwapInOutScheduler(const Context &h_ctx,
                                       const Context &d_ctx,
                                       const size_t max,
                                       const size_t prefetch_max,
                                       const bool save_host_mem,
                                       const bool save_host_mem_no_abort)
  : host_ctx(h_ctx), device_ctx(d_ctx), max_bytes(max),
    max_prefetch_bytes(prefetch_max == 0 ? max * 1.5 : prefetch_max),
    cast_prefetch(save_host_mem),
    cast_prefetch_no_abort(save_host_mem_no_abort),
    sa_callback([&](SyncedArrayPtr saptr, 
                    const SyncedArrayCallbackTag sa_tag,
                    const dtypes dtype,
                    const Context &ctx,
                    const bool write_only,
                    const bool first_creation) {
      // Set SyncedArrayCallback function for first iteration
      sa_callback_recorder(saptr, sa_tag, dtype, ctx, 
                           write_only, first_creation);}) {
    // Create non blocking streams for data transfer
    BackendUtils::create_lms_streams(d_ctx);
}


// Destructor
SwapInOutScheduler::~SwapInOutScheduler() {}


// User interface to start a scheduling code block
void SwapInOutScheduler::start_scheduling() {
  // Init
  order_idx = 0;
  func_idx = 0;
  wrong_ordered.clear();
  precleared.clear();
  set_sa_callback(); // Set SyncedArrayCallback
}


// User interface to finish a scheduling code block
void SwapInOutScheduler::end_scheduling() {
  unset_sa_callback();  // Unset a SyncedArrayCallback
  
  // Post process of the last function.
  if (first_iter) {
    func_block_ends.push_back(order_idx); // Record the end of a function.
    swap_out_first_iter(); // Swap out the arrays of the last function
    schedule();            // Schedule swap in/out
    said_map.clear();      // Clear variables used in the first iteration
  }
  else {
    if (order_idx < func_block_ends[func_idx]) {
      /* If the number of get/cast/clear in this iteration is less than
         the recorded, reset the index to the recorded start position of
         the next function.
      */
      order_idx = func_block_ends[func_idx];
    }

    run_on_end_schedule();
    func_idx++;
    run_on_beginning_schedule(); // Wait for all swap out here
    swap_out_wrong_order();      // Swap out all disordered arrays
  }
  
  /* Host must wait for the all asynchronous memory manipulation for safety.
     That is because data on CPU memory can be destroyed, for esample,
     by the conflict between
     1. asynchronous swap out
     2. the loads of input data and label for the next iteration.
  */
  BackendUtils::device_synchronize(device_ctx);
  
  // After first itration, synced array callback is replaced.
  sa_callback = [&](SyncedArrayPtr saptr, 
                              const SyncedArrayCallbackTag sa_tag,
                              const dtypes dtype, 
                              const Context &ctx,
                              const bool write_only, 
                              const bool first_creation) {
    sa_callback_tracer(saptr, sa_tag, dtype, ctx, write_only, first_creation);};

  first_iter = false;
}


//----------------------------------------------------------------
//  Pre/post hook of function and update
//----------------------------------------------------------------
void SwapInOutScheduler::pre_function_callback(const CgFunctionPtr &ptr) {
  pre_callback();
}

void SwapInOutScheduler::post_function_callback(const CgFunctionPtr &ptr) {
  post_callback();
}

void SwapInOutScheduler::pre_update_callback() {
  pre_callback();
}

void SwapInOutScheduler::post_update_callback() {
  post_callback();
}


//----------------------------------------------------------------
//  Scheduler
//----------------------------------------------------------------
void SwapInOutScheduler::collect_info_about_dtype_conversion
(unordered_map<unsigned int, bool>& type_converted) {
  unordered_map<unsigned int, unordered_map<dtypes, int>> counter;

  for (const auto& r : order) {
    counter[r.said][r.dtype] = 1;
  }

  for (const auto& c : counter) {
    if (c.second.size() > 1) {
      type_converted[c.first] = true;
    }
  }
}


void SwapInOutScheduler::schedule() {
  reconfirm_first_creation();

  unordered_map<unsigned int, bool> type_converted;
  collect_info_about_dtype_conversion(type_converted);

  bool do_reschedule = false;
  auto last_function = func_block_ends.size();

  // Using for prefetch cancel
  vector<unsigned int> prefetch_stopper(order.size(), 1);

  do {
    // Reset schedules
    beginning_schedules.clear();
    end_schedules.clear();
    beginning_schedules.resize(last_function + 1); // +1 is for the last "wait for all"
    end_schedules.resize(last_function);
    ScheduleParams params;

    // Preclear schedule will be used in swap-out schedule.
    unordered_map<unsigned int, vector<pair<RecType*, bool>>> clear_info;
    schedule_preclear(clear_info);

    // Calculate used GPU memory before forward starts, and swap out if necessary.
    calc_mem_usage_before_forward(params);
    schedule_swap_out(params, clear_info);

    // Forward, backward, update
    for (params.fid = 1; params.fid < last_function; params.fid++) {
      schedule_swap_in(params, prefetch_stopper, type_converted);

      do_reschedule =
        reserve_unprefetched_memory(params, prefetch_stopper);

      if (do_reschedule) {
        break;
      }

      if (params.head < func_block_ends[params.fid]) {
        NBLA_ERROR(error_code::memory,
          "Some arrays were not prefetched probably due to out of GPU memory.");
      }

      schedule_swap_out(params, clear_info);
    }

    if (do_reschedule) {
      continue;
    }

    schedule_wait_for_all_swap_out(params);
    determine_which_is_host_func();

    // The end of schedule
    do_reschedule = false;

    // Debug
    #if SWAPINOUTSCHEDULER_DEBUG
    for (const auto& s : params.sa_states) {
      for (const auto& elem : s.second) {
        if (elem.second.state != ArrayStateTag::CLEARED &&
            elem.second.state != ArrayStateTag::OUT_WAITED) {
          NBLA_ERROR(error_code::unclassified, "Invalid ArrayState at the end");
        }
      }
    }
    if (params.prefetch_bytes != 0) {
      NBLA_ERROR(error_code::unclassified, "prefetch_bytes != 0");
    }
    if (params.swap_in_bytes != 0) {
      NBLA_ERROR(error_code::unclassified, "swap_in_bytes != 0");
    }
    if (params.swap_out_bytes != 0) {
      NBLA_ERROR(error_code::unclassified, "swap_out_bytes != 0");
    }
    if (params.head != order.size()) {
      NBLA_ERROR(error_code::unclassified, "head != order.size()");
    }
    if (params.tail != order.size()) {
      NBLA_ERROR(error_code::unclassified, "tail != order.size()");
    }
    #endif
  }  while (do_reschedule);
}


/* Recorded array creations could be fake in the first iteration.
   This method confirms true those first creation.
*/
void SwapInOutScheduler::reconfirm_first_creation() {
  unordered_map<unsigned int, bool> cleared;

  for (auto& r : order) {
    if (r.tag == RecTag::CLEAR) {
      cleared[r.said] = true;
    }
    else {
      if (auto p = r.sawptr.lock()) {
        // If an array is firstly created in the first iteration
        // and it is not cleared at the end, then in the next iteration
        // the array would be alive and not be newly created.
        // This is fake.
        // However, clear() can destroy the alive array even in the next
        // iteration. Therefore the array creation just before clear()
        // is true. This situation could occur after zero() or fill().
        if (r.first_creation && p->get_num_arrays() > 0 && !cleared[r.said]) {
          r.first_creation = false;
        }
      }
    }
  }
}


bool SwapInOutScheduler::no_data_transfer(const RecType* r) {
  return (r->write_only_cast || r->first_creation);
}


int SwapInOutScheduler::
accumulate_counts(const unordered_map<dtypes, ArrayState>& count_map) {
  return accumulate(count_map.begin(), count_map.end(), 0,
    [](int value, const unordered_map<dtypes, ArrayState>::value_type& p)
    { return value + p.second.count; });
}


void SwapInOutScheduler::
backtrack_with_prefetch_cancel(ScheduleParams& params,
                               vector<unsigned int>& prefetch_stopper,
                               const size_t unprefetched_bytes,                               
                               size_t available_bytes) {
  auto back_head = params.head;
  auto back_sa_states = params.sa_states;

  while (back_head >= func_block_ends[params.fid]) {
    back_head--; // decrement first because head indicates the next prefetch array.
    RecType *r = &order[back_head];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }

    if (accumulate_counts(back_sa_states[r->said]) == 1) {
      // Release memory
      for (auto elem : back_sa_states[r->said]) {
        if (elem.second.state == ArrayStateTag::IN) {
          available_bytes += r->size * sizeof_dtype(elem.first);
        }
      }
      prefetch_stopper[back_head] = params.fid + 1;
    }

    if (available_bytes >= unprefetched_bytes) {
      // Ccanceled enough.
      break;
    }

    back_sa_states[r->said][r->dtype].count--;
  }

  if (available_bytes < unprefetched_bytes) {
    NBLA_ERROR(error_code::memory, "A function is out of memory.");
  }
}


bool SwapInOutScheduler::
reserve_unprefetched_memory(ScheduleParams& params,
                            vector<unsigned int>& prefetch_stopper) {
  // Count unprefetched bytes only once
  unordered_map<unsigned int, 
                unordered_map<dtypes, size_t>> uniq_unprefetched_bytes;

  for (auto i = func_block_ends[params.fid - 1]; 
            i < func_block_ends[params.fid]; i++) {
    RecType *r = &order[i];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }

    if (params.sa_states[r->said][r->dtype].state
        == ArrayStateTag::UNPREFETCHED) {
      uniq_unprefetched_bytes[r->said][r->dtype]
        = r->size * sizeof_dtype(r->dtype);
    }
  }

  size_t unprefetched_bytes = 0;
  for (const auto& sa_bytes : uniq_unprefetched_bytes) {
    for (const auto bytes : sa_bytes.second) {
      unprefetched_bytes += bytes.second;
    }
  }

  while (max_bytes - params.swap_in_bytes - params.swap_out_bytes
                                        < unprefetched_bytes) {
    if (params.tail == func_block_ends[params.fid - 1]) {
      // Out of memory, do backtrack with prefetch cancel and reschedule.
      auto available_bytes
        = max_bytes - params.swap_in_bytes - params.swap_out_bytes;
      backtrack_with_prefetch_cancel(params, prefetch_stopper, 
                                     unprefetched_bytes, available_bytes);
      return true;
    }

    // Wait for swap out and release memory
    schedule_wait_for_swap_out_impl(params);
  }

  // Memory for unprefetched arrays became available.
  params.swap_in_bytes += unprefetched_bytes;

  for (auto i = func_block_ends[params.fid - 1];
            i < func_block_ends[params.fid]; i++) {
    RecType *r = &order[i];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }
    
    // UNPREFETCHED to IN
    if (params.sa_states[r->said][r->dtype].state
        == ArrayStateTag::UNPREFETCHED) {
      params.sa_states[r->said][r->dtype].state = ArrayStateTag::IN;
    }
  }

  return false;
}


void SwapInOutScheduler::determine_which_is_host_func() {
  for (int fid = 0; fid < func_block_ends.size(); fid++) {
    bool host_func = false;

    for (size_t i = (fid == 0 ? 0 : func_block_ends[fid - 1]);
      i < func_block_ends[fid]; i++) {
      RecType *r = &order[i];

      if (r->tag == RecTag::CLEAR) {
        continue;
      }

      if (context_checker(r->ctx, host_ctx)) {
        host_func = true;
      }
    }

    is_host_func.push_back(host_func);
  }
}


void SwapInOutScheduler::
calc_mem_usage_before_forward(ScheduleParams& params) {
  while (params.head < func_block_ends[0]) {
    RecType *r = &order[params.head];

    if (r->tag == RecTag::CLEAR) {
      params.head++;
      continue;
    }

    if (!context_checker(r->ctx, device_ctx) &&
        !context_checker(r->ctx, host_ctx)) {
      NBLA_ERROR(error_code::type,
                 "Unsupported array type: " + r->ctx.array_class);
    }

    // All fetches were already done. Just calculate memory size.
    auto array_bytes = r->size * sizeof_dtype(r->dtype);

    // First fetch
    if (params.sa_states[r->said][r->dtype].count == 0) {
      params.swap_in_bytes += array_bytes;
      params.prefetch_bytes += array_bytes;

      // CLEARED to IN
      if (params.sa_states[r->said][r->dtype].state != ArrayStateTag::CLEARED) {
        NBLA_ERROR(error_code::type, "CLEARED");
      }
      params.sa_states[r->said][r->dtype].state = ArrayStateTag::IN;
    }

    // Increment the number of the same SyncedArray in the queue.
    params.sa_states[r->said][r->dtype].count++;
    params.head++; // Move on the head of the queue
  }
}

void SwapInOutScheduler::
cancel_swap_out(const RecType *r, ScheduleParams& params) {
  [&] { // This lambda enables us to break double for-loop.
    for (size_t i = end_schedules.size(); i-- > 0;) {
      // Search for the last swap out and cancel it
      for (size_t j = end_schedules[i].size(); j-- > 0;) {
        if (end_schedules[i][j].r->said == r->said &&
          end_schedules[i][j].tag == ScheduleTag::SWAP_OUT) {
          end_schedules[i].erase(end_schedules[i].begin() + j);
          return;
        }
      }
    }
  } ();

  size_t bytes = 0;
  for (auto& elem : params.sa_states[r->said]) {
    if (elem.second.state == ArrayStateTag::OUT) {
      bytes += r->size * sizeof_dtype(elem.first);
      elem.second.state = ArrayStateTag::IN; // OUT to IN
    }
  }

  params.swap_out_bytes -= bytes;
  params.swap_in_bytes += bytes;
  params.prefetch_bytes += bytes;
}


bool SwapInOutScheduler::
free_memory_to_prefetch(ScheduleParams& params, const size_t array_bytes) {
  bool no_memory = false;

  while (params.swap_in_bytes + params.swap_out_bytes + array_bytes 
                                                           > max_bytes) {
    if (params.tail == func_block_ends[params.fid - 1]) {
      no_memory = true;
      break;
    }

    // Out of memory. Wait for swap out and release memory
    schedule_wait_for_swap_out_impl(params);
  }
  
  return no_memory;
}


void SwapInOutScheduler::
schedule_swap_in(ScheduleParams& params,
                 const vector<unsigned int> prefetch_stopper,
                 unordered_map<unsigned int, bool>& type_converted) {
  while (params.head < order.size()) {
    RecType *r = &order[params.head];

    if (r->tag == RecTag::CLEAR) {
      params.head++;
      continue;
    } 

    if (!context_checker(r->ctx, device_ctx) &&
        !context_checker(r->ctx, host_ctx)) {
      NBLA_ERROR(error_code::type,
                 "Unsupported array type: " + r->ctx.array_class);
    }

    // Prefetch must be stopped to avoid out-of-memory in the future.
    if (prefetch_stopper[params.head] > params.fid) break;

    if (params.sa_states[r->said][r->dtype].count == 0) {
      // The array is firstly appeared in the queue.
      const auto array_bytes = r->size * sizeof_dtype(r->dtype);

      // Out of prefetch memory
      if (max_prefetch_bytes < params.prefetch_bytes + array_bytes ||
          max_bytes < params.swap_in_bytes + array_bytes) {
        break;
      }

      if (context_checker(r->ctx, device_ctx)) {
        if (params.sa_states[r->said][r->dtype].state == ArrayStateTag::OUT) {
          // Swap out cancel
          cancel_swap_out(r, params);
        }
        else if (no_data_transfer(r)) { // Unprefetch due to no data transfer
          params.prefetch_bytes += array_bytes;

          // CLEARED -> UNPREFETCHED
          params.sa_states[r->said][r->dtype].state = ArrayStateTag::UNPREFETCHED;
        }
        else { // Prefetch
          // Free memory to prefetch the next array
          free_memory_to_prefetch(params, array_bytes);

          if (!cast_prefetch || type_converted[r->said]) {
            beginning_schedules[params.fid]
              .push_back(ScheduleType(ScheduleTag::SWAP_IN_GET, r));
          }
          else {
            beginning_schedules[params.fid]
              .push_back(ScheduleType(ScheduleTag::SWAP_IN_CAST, r));
          }

          params.swap_in_bytes += array_bytes; // Increase memory usage
          params.prefetch_bytes += array_bytes;

          // CLEARED or OUT_WAITED to IN
          params.sa_states[r->said][r->dtype].state = ArrayStateTag::IN;
        }
      }
      else { // Host array
        // Not prefetch and synchronous fetch in functions
        params.swap_in_bytes += array_bytes;
        params.prefetch_bytes += array_bytes;

        // CLEARED or OUT_WAITED to IN
        params.sa_states[r->said][r->dtype].state = ArrayStateTag::IN;
      }
    }

    // Increment the number of the same SyncedArray in the queue.
    params.sa_states[r->said][r->dtype].count++;
    params.head++; // Move on the head of the queue
  }
}


void SwapInOutScheduler::
schedule_swap_out(ScheduleParams& params,
                  unordered_map<unsigned int, 
                     vector<pair<RecType*, bool>>>& clear_info) {
  for (size_t i = (params.fid == 0 ? 0 : func_block_ends[params.fid - 1]);
              i < func_block_ends[params.fid];
              i++) {
    RecType *r = &order[i];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }

    if (!context_checker(r->ctx, device_ctx) &&
        !context_checker(r->ctx, host_ctx)) {
      NBLA_ERROR(error_code::type,
                 "Unsupported array type: " + r->ctx.array_class);
    }

    if (accumulate_counts(params.sa_states[r->said]) == 1) {
      // An array is swapped out when the same array is no longer
      // in the queue.
      auto do_preclear = std::find_if(clear_info[params.fid].begin(),
                                      clear_info[params.fid].end(),
        [&](pair<RecType*, bool> x) { return x.first == r; });

      if (do_preclear != clear_info[params.fid].end()) { // Cleared
        for (auto& elem : params.sa_states[r->said]) { // Any states to CLEARED
          if (elem.second.state == ArrayStateTag::IN) {
            auto array_bytes = r->size * sizeof_dtype(elem.first);
            params.swap_in_bytes -= array_bytes;
            params.prefetch_bytes -= array_bytes;
          }

          elem.second.state = ArrayStateTag::CLEARED;
        }

        if (do_preclear->second) {
          end_schedules[params.fid]
            .push_back(ScheduleType(ScheduleTag::PRECLEAR, r));
        }
      }
      else { // Not precleared, Swap out
        end_schedules[params.fid]
          .push_back(ScheduleType(ScheduleTag::SWAP_OUT, r));
        
        // Transfer memory usage of all types
        for (auto& elem : params.sa_states[r->said]) {
          if (elem.second.state == ArrayStateTag::IN) {
            auto array_bytes = r->size * sizeof_dtype(elem.first);
            params.swap_out_bytes += array_bytes;
            params.swap_in_bytes -= array_bytes;
            params.prefetch_bytes -= array_bytes;
            elem.second.state = ArrayStateTag::OUT; // IN to OUT
          }
        }
      }
    }

    // Decrease the counts of a used array in the queue.
    params.sa_states[r->said][r->dtype].count--;
  }
}


void SwapInOutScheduler::
schedule_wait_for_all_swap_out(ScheduleParams& params) {
  // When out of memory, wait for finishing swap out.
  while (params.tail < order.size()) {
    schedule_wait_for_swap_out_impl(params);
  }
}


void SwapInOutScheduler::
schedule_wait_for_swap_out_impl(ScheduleParams& params) {
  RecType *r = &order[params.tail++];

  if (params.sa_states[r->said][r->dtype].state == ArrayStateTag::OUT) {
    // Not canceled swap out
    // Wait for finishing swap out and release the source array of memory copy.
    beginning_schedules[params.fid]
      .push_back(ScheduleType(ScheduleTag::WAIT, r));

    // Decrease memory usage
    size_t bytes = 0;
    for (auto& elem : params.sa_states[r->said]) {
      if (elem.second.state == ArrayStateTag::OUT) {
        bytes += r->size * sizeof_dtype(elem.first);
        elem.second.state = ArrayStateTag::OUT_WAITED; // OUT to OUT_WAITED
      }
    }
    params.swap_out_bytes -= bytes;
  }
}


void SwapInOutScheduler::
schedule_preclear(unordered_map<unsigned int, 
                                vector<pair<RecType*, bool>>>& clear_info) {
  unordered_map<unsigned int, pair<bool, int>> clear_trace;
  unordered_map<unsigned int, bool> clear_last;

  // fid == 0 is before the first pre-function hook. No chance to preclear.
  for (int fid = func_block_ends.size() - 1; fid > 1; fid--) {
    for (int i = func_block_ends[fid] - 1; i >= func_block_ends[fid - 1]; i--) {
      RecType *r = &order[i];

      if (r->tag == RecTag::CLEAR) {
        clear_trace[r->said] = {true, fid};

        if (clear_last.find(r->said) == clear_last.end()) {
          clear_last[r->said] = true;
        }
      }
      else {
        if (clear_last.find(r->said) == clear_last.end()) {
          clear_last[r->said] = false;
        }

        if (clear_trace[r->said].first) { // get/cast just before clear
          if (fid < clear_trace[r->said].second) {
            // Preclear
            clear_info[fid].push_back({r, true});
          }
          else if (clear_last.at(r->said)) {
            // Do not preclear because clear before preclear.
            clear_info[fid].push_back({r, false});
          }

          clear_trace[r->said] = {false, -1};
        }
      }
    }

    clear_last.clear();
  }
}


//----------------------------------------------------------------
//  Execute swap in/out
//----------------------------------------------------------------
// Common implementation of pre callback
void SwapInOutScheduler::pre_callback() {
  unset_sa_callback(); // Avoid unnecessary record and trace

  if (first_iter) {
    // Record the end of a function.
    func_block_ends.push_back(order_idx);

    // Swap out and preclear the arrays used in the previous function.
    swap_out_first_iter();

    func_idx++;
  }
  else {
    if (order_idx < func_block_ends[func_idx]) {
      /* If the number of get/cast/clear in this iteration is less than
      the recorded, reset the index to the recorded start position of
      the next function.
      */
      order_idx = func_block_ends[func_idx];
    }

    run_on_end_schedule();
    func_idx++;
    run_on_beginning_schedule();

    if (is_host_func[func_idx]) {
      BackendUtils::default_stream_synchronize(device_ctx);
    }
  }

  set_sa_callback(); // Restart record or trace
}


// Common implementation of post callback
void SwapInOutScheduler::post_callback() {}


void SwapInOutScheduler::run(const ScheduleType& s) {
  if (s.tag == ScheduleTag::SWAP_IN_GET) {
    if (auto p = s.r->sawptr.lock()) {
      p->get(s.r->dtype, device_ctx, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
    }
  }
  else if (s.tag == ScheduleTag::SWAP_IN_CAST) {
    if (auto p = s.r->sawptr.lock()) {
      p->cast(s.r->dtype, device_ctx, false, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
    }
  }
  else if (s.tag == ScheduleTag::SWAP_OUT) {
    // Swap out and preclear the arrays used in the previous function.
    if (auto p = s.r->sawptr.lock()) {
      if (is_not_cleared_yet(p)) {
        p->cast(p->dtype(), host_ctx, false, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
      }
    }
  }
  else if (s.tag == ScheduleTag::WAIT) {
    auto p = s.r->sawptr.lock();

    if (p && p->head_array_class() == host_ctx.array_class &&
        is_not_cleared_yet(p)) {
      p->get(p->dtype(), host_ctx, AsyncFlag::UNSAFE);
    }
  }
  else if (s.tag == ScheduleTag::PRECLEAR) {
    if (auto p = s.r->sawptr.lock()) {
      p->clear();
      precleared[p] = true;
    }
  }
}

void SwapInOutScheduler::run_on_beginning_schedule() {
  for (const auto& s : beginning_schedules[func_idx]) { run(s); }
}


void SwapInOutScheduler::run_on_end_schedule() {
  for (const auto& s : end_schedules[func_idx]) { run(s); }
}


//----------------------------------------------------------------
//  First iteration
//----------------------------------------------------------------
void SwapInOutScheduler::swap_out_first_iter() {
  // In the first iteration, arrays used in a function are always swapped out.
  const int start_idx = func_idx == 0 ? 0 : func_block_ends[func_idx - 1];

  for (int i = start_idx; i < func_block_ends[func_idx]; i++) {
    RecType *r = &order[i];
    if (r->tag == RecTag::CLEAR) continue;

    if (context_checker(r->ctx, device_ctx)) {
      auto p = r->sawptr.lock();

      if (p && is_not_cleared_yet(p)) {
        // The array is not cleared yet. Swap it out.
        p->cast(p->dtype(), host_ctx, false);
      }
    }
  }
}


void SwapInOutScheduler::swap_out_wrong_order() {
  for (int i = 0; i < wrong_ordered.size(); i++) {
    RecType *r = &wrong_ordered[i];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }

    if (context_checker(r->ctx, device_ctx)) {
      auto p = r->sawptr.lock();

      if (p && is_not_cleared_yet(p)) {
        // Swap out the array SYNCRONOUSLY because device synchronize will be 
        // called just after this.
        p->cast(r->dtype, host_ctx, false);
      }
    }
    else if (!context_checker(r->ctx, host_ctx)) {
      // Function used an array on an uncertain device
      NBLA_ERROR(error_code::type,
                 "Unsupported array class: " + r->ctx.array_class);
    }
  }
}


//----------------------------------------------------------------
//  SyncedArrayCallback function
//----------------------------------------------------------------
void SwapInOutScheduler::set_sa_callback() {
  SingletonManager::get<SyncedArrayCallback>()->set_callback_func(sa_callback);
}


void SwapInOutScheduler::unset_sa_callback() {
  SingletonManager::get<SyncedArrayCallback>()->set_callback_func(nullptr);
}


// SyncedArrayCallback function to record the order
void SwapInOutScheduler::
sa_callback_recorder(SyncedArrayPtr saptr, const SyncedArrayCallbackTag sa_tag,
                     const dtypes dtype, const Context &ctx,
                     const bool write_only, const bool first_creation) {
  // Define SyncedArray ID
  if (said_map.find(saptr) == said_map.end()) {
    said_map[saptr] = static_cast<unsigned int>(said_map.size());
  }
  auto said = said_map.at(saptr);

  // Record the order
  auto tag = convert_tag(sa_tag, write_only);
  order.push_back(RecType{tag, said, saptr, saptr->size(), dtype, ctx,
                          write_only, first_creation});
  said_to_order_idx[said].push_back(order_idx);
  order_idx++;
}


// SyncedArrayCallback function to trace the recorded order
void SwapInOutScheduler::
sa_callback_tracer(SyncedArrayPtr saptr, const SyncedArrayCallbackTag sa_tag,
                   const dtypes dtype, const Context &ctx,
                   const bool write_only, const bool first_creation) {
  auto tag = convert_tag(sa_tag, write_only);

  // Unexpected get/cast happens after preclear destroys data. Abort.
  if (precleared[saptr]) {
    if (tag == RecTag::CLEAR) { // Actual clear. it is Ok.
      precleared[saptr] = false;
    }
    else { // Unexpected get/cast. Abort.
      NBLA_ERROR(error_code::unclassified,
                 "Unexpected get or cast appears after preclear.");
    }
  }

  // Check if the on-going iteration follows the recorded order.
  auto r = order[order_idx];
  auto rec_saptr = r.sawptr.lock();

  // Case 1: Over the current function block.
  if (order_idx >= func_block_ends[func_idx]) {
    wrong_ordered.push_back({tag, 0, saptr, saptr->size(), dtype, ctx,
                             false, false});
  }
  // Case 2: SyncedArray replacement in the same order
  else if (r.sawptr.expired() && // "expired" implies the replacement.
           tag == r.tag &&
           saptr->size() == r.size &&
           dtype == r.dtype &&
           ctx.array_class == r.ctx.array_class) {
    // Replace all recorded SyncedArray with new one
    for (auto& i : said_to_order_idx[r.said]) {
      order[i].sawptr = saptr;
    }
  }
  // Case 3: Different entry
  else if (tag != r.tag || 
           saptr != rec_saptr ||
           saptr->size() != r.size ||
           dtype != r.dtype ||
           ctx.array_class != r.ctx.array_class) {
   wrong_ordered.push_back({tag, 0, saptr, saptr->size(), dtype, ctx,
                            false, false});
  }

  order_idx++;
}


SwapInOutScheduler::RecTag
SwapInOutScheduler::convert_tag(const SyncedArrayCallbackTag sa_tag,
                            const bool write_only) {
  if (sa_tag == SyncedArrayCallbackTag::GET) {
    return RecTag::GETCAST;
  }
  else if (sa_tag == SyncedArrayCallbackTag::CAST) {
    return RecTag::GETCAST;
  }
  else if (sa_tag == SyncedArrayCallbackTag::CLEAR) {
    return RecTag::CLEAR;
  }

  NBLA_ERROR(error_code::type, "Unsupported SyncedArrayCallbackTag");
}
}
