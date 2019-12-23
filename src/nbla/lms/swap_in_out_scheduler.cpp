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
#include <nbla/backend_registry.hpp>
#include <nbla/singleton_manager.hpp>


namespace nbla {

using std::accumulate;


SwapInOutScheduler::SwapInOutScheduler(const Context &h_ctx,
                                       const Context &d_ctx,
                                       const size_t bytes)
  : host_ctx(h_ctx), device_ctx(d_ctx),
    // Balancing the maximum GPU memory size for swap out in half
    max_bytes_swap_in(bytes), max_bytes_swap_out(bytes / 2),
    // Set SyncedArrayCallback function for first iteration
    synced_array_callback(
      [&](SyncedArrayPtr saptr, const SyncedArrayCallbackTag func_name,
          const dtypes dtype, const Context &ctx, const bool write_only) {
       synced_array_callback_recorder(saptr, func_name, dtype, 
                                      ctx, write_only); }) {
    // Create non blocking streams for data transfer
    BackendUtils::create_lms_streams(d_ctx);
}


SwapInOutScheduler::~SwapInOutScheduler() {}


void SwapInOutScheduler::start_scheduling() {
  init();
}


void SwapInOutScheduler::end_scheduling() {
  finalize();
}


void SwapInOutScheduler::init() {
  tail = 0;
  used_bytes_swap_out = 0;
  order_idx = 0;
  func_idx = 0;
  wrong_ordered.clear();
  precleared.clear();
  synced_array_id_mapper.clear();

  set_synced_array_callback();
}


void SwapInOutScheduler::finalize() {
  unset_synced_array_callback();
  
  swap_out_step(); // Post process of the last function.

  // Swap out all arrays out of the recorded order.
  swap_out_wrong_order();

  // Schedule at the end of first iteration
  if (first_iter) {
    // Wait for swaping out all arrays and clean GPU memory for the next iteration.
    wait_for_all_swap_out_first_iter();

    if (used_bytes_swap_out != 0) {
      NBLA_ERROR(error_code::unclassified, "used_bytes_swap_out != 0");
    }

    if (tail != order.size()) {
      NBLA_ERROR(error_code::unclassified, "tail != order.size()");
    }

    init();
    schedule();
  }
  else {
    // Wait for swaping out all arrays and clean GPU memory for the next iteration.
    wait_for_all_swap_out_scheduled();
  }

  /* Host must wait for the all asynchronous memory manipulation for safety.
     That is because the destruction of data on CPU memory can occur
     by overriting CPU memory conflictedly between
     1. asynchronous swap out
     2. the loads of input data and label for the next iteration.
  */
  BackendUtils::device_synchronize(device_ctx);
  
  // After first itration, synced array callback is replaced.
  synced_array_callback = 
    [&](SyncedArrayPtr saptr, const SyncedArrayCallbackTag func_name,
        const dtypes dtype, const Context &ctx, const bool write_only) {
    synced_array_callback_tracer(saptr, func_name, dtype, ctx, write_only); };

  first_iter = false;
}


//----------------------------------------------------------------
//  Pre/post hook of function and update
//----------------------------------------------------------------

void SwapInOutScheduler::pre_function_callback(const CgFunctionPtr &ptr) {
  pre_callback();
}

void SwapInOutScheduler::post_function_callback(const CgFunctionPtr &ptr) {}

void SwapInOutScheduler::pre_update_callback() {
  pre_callback();
}

void SwapInOutScheduler::post_update_callback() {}


//----------------------------------------------------------------
//  Scheduler
//----------------------------------------------------------------

void SwapInOutScheduler::schedule() {
  schedule_preclear(); // Preclear schedule will be used in swap-out schedule.

  /* This counts the number of same arrays in the queue.
     If count of an array > 0, no need to fetch the same array again.
     If count of an array > 1, no need to swap out the array because it is
     planed to be used in the queue.
  */
  SyncedArrayCountsInQueue synced_array_counts;

  int head = 0;
  size_t used_bytes_swap_in = 0;
  auto last_function = func_block_ends.size();
  unordered_map<unsigned int, bool> host_uses_this_synced_array;

  // They are used to remove uneccesary swap-out
  unordered_map<unsigned int, bool> swapped_out;
  unordered_map<unsigned int, RecType*> swapped_out_r;

  // Virtually iterate all layer functions and solver update
  int fid = 0;

  // Calculate used GPU memory before forward starts,
  // and swap out if necessary.
  detect_swap_in_before_forward(head, used_bytes_swap_in,
                                synced_array_counts);
  swap_out_schedule[fid] = schedule_swap_out(used_bytes_swap_in,
                                             synced_array_counts, fid,
                                             swapped_out, swapped_out_r);
  wait_schedule[fid] = schedule_wait_for_swap_out(swapped_out, swapped_out_r);

  // Forward, backward, update
  for (fid = 1; fid < last_function; fid++) {
    swap_in_schedule[fid] = schedule_swap_in(head, fid, used_bytes_swap_in, 
                                             synced_array_counts,
                                             host_uses_this_synced_array,
                                             swapped_out, swapped_out_r);

    if (head < func_block_ends[fid]) {
      NBLA_ERROR(error_code::memory,
        "Some arrays were not prefetched probably due to out of GPU memory. ");
    }

    swap_out_schedule[fid] = schedule_swap_out(used_bytes_swap_in, 
                                               synced_array_counts, fid,
                                               swapped_out, swapped_out_r);

    wait_schedule[fid] = schedule_wait_for_swap_out(swapped_out, swapped_out_r);
  }

  wait_all_schedule = schedule_wait_for_all_swap_out(swapped_out, swapped_out_r);

  check_which_is_host_func();

  // Debug
  if (used_bytes_swap_in != 0) {
    NBLA_ERROR(error_code::unclassified, "used_bytes_swap_in != 0");
  }
  if (used_bytes_swap_out != 0) {
    NBLA_ERROR(error_code::unclassified, "used_bytes_swap_out != 0");
  }
  if (head != order.size()) {
    NBLA_ERROR(error_code::unclassified, "head != order.size()");
  }
  if (tail != order.size()) {
    NBLA_ERROR(error_code::unclassified, "tail != order.size()");
  }
}


int accumulate_counts(const unordered_map<dtypes, int>& count_map) {
  return accumulate(count_map.begin(), count_map.end(), 0,
    [](int value, const unordered_map<dtypes, int>::value_type& p)
    { return value + p.second; });
}


void SwapInOutScheduler::check_which_is_host_func() {
  for (int fid = 0; fid < func_block_ends.size(); fid++) {
    bool host_func = false;

    for (size_t i = (fid == 0 ? 0 : func_block_ends[fid - 1]);
      i < func_block_ends[fid]; i++) {
      RecType& r = order[i];

      if (r.tag == RecTag::CLEAR) {
        continue;
      }

      if (r.ctx.array_class == host_ctx.array_class) {
        host_func = true;
      }
    }

    is_host_func.push_back(host_func);
  }
}


void SwapInOutScheduler::
detect_swap_in_before_forward(int& head, size_t& used_bytes_swap_in,
                              SyncedArrayCountsInQueue& synced_array_counts) {
  while (head < func_block_ends[0]) {
    RecType& r = order[head];

    if (r.tag == RecTag::CLEAR) {
      head++;
      continue;
    }

    if (r.ctx.array_class == device_ctx.array_class) {
      // All fetches were already done. Just calculate memory size.
      auto array_bytes = r.size * sizeof_dtype(r.dtype);

      // First fetch
      if (synced_array_counts[r.synced_array_id][r.dtype] == 0) {
        used_bytes_swap_in += array_bytes;
      }

      // Increment the number of the same SyncedArray in the queue.
      synced_array_counts[r.synced_array_id][r.dtype]++;
      head++; // Move on the head of the queue
    }
    else if (r.ctx.array_class == host_ctx.array_class) {
      // Because func_idx == 0 means all get/cast finished already
      // the host process must be finished.

      head++; // Move on the head of the queue
    }
    else {
      // Get/cast of an array on an uncertain device
      NBLA_ERROR(error_code::type,
                 "Unsupported array type: " + r.ctx.array_class);
    }
  }
}


SwapInOutScheduler::ScheduleType
SwapInOutScheduler::
schedule_swap_in(int& head, const int fid, size_t& used_bytes_swap_in,
                 SyncedArrayCountsInQueue& synced_array_counts,
                 unordered_map<unsigned int, bool>& host_uses_this_synced_array,
                 unordered_map<unsigned int, bool>& swapped_out,
                 unordered_map<unsigned int, RecType*>& swapped_out_r) {
  SwapInOutScheduler::ScheduleType schedule;

  while (head < order.size()) {
    RecType& r = order[head];

    if (r.tag == RecTag::CLEAR) {
      head++;
      continue;
    }
    
    if (r.ctx.array_class == device_ctx.array_class) {
      auto next_array_bytes = r.size * sizeof_dtype(r.dtype);

      if (used_bytes_swap_in + next_array_bytes
          > max_bytes_swap_in - used_bytes_swap_out) {
        break; // Out of memory. Stop fetching.
      }

      // Fetch
      if (synced_array_counts[r.synced_array_id][r.dtype] == 0) {
        // The array is firstly appeared in the queue.

        if (!host_uses_this_synced_array[r.synced_array_id]) {
          schedule.push_back(r);

          // If the array was previously swapped out,
          // the memcpy was waited for by swap in.
          if (swapped_out[r.synced_array_id]) {
            auto rptr = swapped_out_r[r.synced_array_id];
            
            // The array is used before swap out is completed.
            // It is not required to swap out the array.
            rptr->no_need_swap_out = true;

            // Remove memory size from swap-out memory
            rptr->swapped_out = false;
            used_bytes_swap_out -= rptr->swapped_out_bytes;
            rptr->swapped_out_bytes = 0;

            // reset flag
            swapped_out[r.synced_array_id] = false;
            swapped_out_r[r.synced_array_id] = nullptr;
          }
        }

        // Increase memory usage
        used_bytes_swap_in += next_array_bytes;
      }

      // Increment the number of the same SyncedArray in the queue.
      synced_array_counts[r.synced_array_id][r.dtype]++;
      head++; // Move on the head of the queue
    }
    else if (r.ctx.array_class == host_ctx.array_class) {
      // No need swap-in (prefetch) to CPU. The array will be gotten/casted 
      // synchronously by the function itself. 
      // Stop prefetch these type of arrays.
      if (fid > 0) {
        // Because func_idx == 0 means all get/cast finished already
        // the host process must be finished.
        host_uses_this_synced_array[r.synced_array_id] = true;
      }

      // The arrray comes on to host by swap out.
      if (swapped_out[r.synced_array_id]) {
        // reset flag
        swapped_out[r.synced_array_id] = false;
        swapped_out_r[r.synced_array_id] = nullptr;
      }

      head++;
    }
    else {
      // Get/cast of an array on an uncertain device
      NBLA_ERROR(error_code::type, 
                 "Unsupported array type: " + r.ctx.array_class);
    }
  }

  return schedule;
}


SwapInOutScheduler::ScheduleType
SwapInOutScheduler::
schedule_swap_out(size_t& used_bytes_swap_in, 
                  SyncedArrayCountsInQueue& synced_array_counts, 
                  const int fid,
                  unordered_map<unsigned int, bool>& swapped_out,
                  unordered_map<unsigned int, RecType*>& swapped_out_r) {
  SwapInOutScheduler::ScheduleType schedule;

  for (size_t i = (fid == 0 ? 0 : func_block_ends[fid - 1]);
              i < func_block_ends[fid];
              i++) {
    RecType& r = order[i];

    if (r.tag == RecTag::CLEAR) {
      continue;
    }

    if (r.ctx.array_class == device_ctx.array_class) { 
      if (accumulate_counts(synced_array_counts[r.synced_array_id]) == 1) {
        // An array is swapped out when the same array is no longer
        // in the queue.
        schedule.push_back(r);

        if (!r.preclear) {
          r.swapped_out = true;
          swapped_out[r.synced_array_id] = true;
          swapped_out_r[r.synced_array_id] = &order[i];
        
          // Transfer memory usage of all types
          r.swapped_out_bytes = 0;

          for (auto it : synced_array_counts[r.synced_array_id]) {
            auto array_bytes = r.size * sizeof_dtype(it.first);
            used_bytes_swap_out += array_bytes;
            r.swapped_out_bytes += array_bytes;
          }
        }

        // Transfer memory usage of all types
        for (auto it : synced_array_counts[r.synced_array_id]) {
          auto array_bytes = r.size * sizeof_dtype(it.first);
          used_bytes_swap_in -= array_bytes;
        }

        // Reset usage
        synced_array_counts[r.synced_array_id].clear();
      }
      else {
        // Decrease the counts of a used array in the queue.
        synced_array_counts[r.synced_array_id][r.dtype]--;
      }
    }
    else if (r.ctx.array_class != host_ctx.array_class) {
      // Get/cast of an array on an uncertain device
      NBLA_ERROR(error_code::type, 
                 "Unsupported array type: " + r.ctx.array_class);
    }
  }

  return schedule;
}


SwapInOutScheduler::ScheduleType
SwapInOutScheduler::schedule_wait_for_swap_out(unordered_map<unsigned int, bool>& swapped_out,
                                               unordered_map<unsigned int, RecType*>& swapped_out_r) {
  SwapInOutScheduler::ScheduleType schedule;

  // When out of memory, wait to finish swap-out and release memory.
  while (used_bytes_swap_out > max_bytes_swap_out) {
    schedule_wait_for_swap_out_impl(schedule, swapped_out, swapped_out_r);
  }

  return schedule;
}


SwapInOutScheduler::ScheduleType
SwapInOutScheduler::schedule_wait_for_all_swap_out(unordered_map<unsigned int, bool>& swapped_out,
                                                   unordered_map<unsigned int, RecType*>& swapped_out_r) {
  SwapInOutScheduler::ScheduleType schedule;

  // When out of memory, wait for finishing swap out.
  while (tail < order.size()) {
    schedule_wait_for_swap_out_impl(schedule, swapped_out, swapped_out_r);
  }

  return schedule;
}


void SwapInOutScheduler::schedule_wait_for_swap_out_impl(
  SwapInOutScheduler::ScheduleType& schedule,
  unordered_map<unsigned int, bool>& swapped_out,
  unordered_map<unsigned int, RecType*>& swapped_out_r) {
  RecType& r = order[tail++];

  if (r.swapped_out) {
    // Wait for finishing swap out and release the source array of memory copy.
    schedule.push_back(r);

    // Decrease memory usage
    r.swapped_out = false;
    used_bytes_swap_out -= r.swapped_out_bytes;
    r.swapped_out_bytes = 0;

    swapped_out[r.synced_array_id] = false;
    swapped_out_r[r.synced_array_id] = nullptr;
  }
}


// For the same SyncedArray, get/cast just before the clear is
// time to preclear it instead of swapping it out.
void SwapInOutScheduler::schedule_preclear() {
  unordered_map<unsigned int, bool> clear_flag;

  for (int i = order.size() - 1; i >= 0; i--) {
    if (order[i].tag == RecTag::CLEAR) {
      clear_flag[order[i].synced_array_id] = true;
    }
    else {
      order[i].preclear = clear_flag[order[i].synced_array_id];
      clear_flag[order[i].synced_array_id] = false;
    }
  }
}



//----------------------------------------------------------------
//  Execution
//----------------------------------------------------------------

// Common implementation of pre callback
void SwapInOutScheduler::pre_callback() {
  unset_synced_array_callback(); // Avoid unnecessary record and trace

  swap_out_step(); // post process of the previous function
  swap_in_step();  // pre process of the next function
  
  if (!first_iter && is_host_func[func_idx]) {
    BackendUtils::default_stream_synchronize(device_ctx);
  }

  set_synced_array_callback(); // Restart record or trace
}


void SwapInOutScheduler::swap_out_step() {
  // Record the end of a function.
  if (first_iter) {
    func_block_ends.push_back(order_idx);
  }

  // Swap out and preclear the arrays used in the previous function.
  swap_out();

  if (order_idx < func_block_ends[func_idx]) {
    /* If the number of get/cast/clear in this iteration is less than
       the recorded, reset the index to the recorded start position of 
       the next function.
     */
    order_idx = func_block_ends[func_idx];
  }
}


void SwapInOutScheduler::swap_in_step() {
  func_idx++;

  if (!first_iter) {
    swap_in(); // Prefetch arrays as possible.
  }
}


// Swap in (prefetch)
void SwapInOutScheduler::swap_in() {
  for (auto r : swap_in_schedule[func_idx]) {
    if (auto p = r.get().sawptr.lock()) {
      p->get(r.get().dtype, device_ctx, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
    }
  }
}

// Swap out
void SwapInOutScheduler::swap_out() {
  // Swap out the arrays used in the previous function
  if (first_iter) {
    swap_out_first_iter();
    wait_for_swap_out_first_iter();
  }
  else {
    swap_out_scheduled();
    wait_for_swap_out_scheduled();
  }
}


// In the first iteration, arrays used in a function are always swapped out.
void SwapInOutScheduler::swap_out_first_iter() {
  // Counts SyncedArrays which were used in the previous function.
  SyncedArrayCountsInQueue synced_array_counts;
  const int start_idx = func_idx == 0 ? 0 : func_block_ends[func_idx - 1];

  for (int i = start_idx; i < func_block_ends[func_idx]; i++) {
    RecType& r = order[i];
    if (r.tag == RecTag::CLEAR) continue;

    if (r.ctx.array_class == device_ctx.array_class) {
      synced_array_counts[r.synced_array_id][r.dtype]++;
    }
    else if (r.ctx.array_class != host_ctx.array_class) {
      // Get/cast of an array on an uncertain device
      NBLA_ERROR(error_code::type, "Unsupported array type: " + r.ctx.array_class);
    }
  }

  // Swap out
  for (int i = start_idx; i < func_block_ends[func_idx]; i++) {
    RecType& r = order[i];
    if (r.tag == RecTag::CLEAR) continue;

    if (r.ctx.array_class == device_ctx.array_class) {
      if (accumulate_counts(synced_array_counts[r.synced_array_id]) == 1) {
        auto p = r.sawptr.lock();

        if (p && p->get_num_arrays() > 0) {
          // The array is not cleared yet. Swap out the array
          p->cast(p->dtype(), host_ctx, false, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
          r.swapped_out = true;

          // Counts memory usage of all types
          r.swapped_out_bytes = 0;

          for (auto it : synced_array_counts[r.synced_array_id]) {
            auto array_bytes = r.size * sizeof_dtype(it.first);
            used_bytes_swap_out += array_bytes;
            r.swapped_out_bytes += array_bytes;
          }
        }

        synced_array_counts[r.synced_array_id].clear();
      }
      else {
        synced_array_counts[r.synced_array_id][r.dtype]--;
      }
    }
  }
}

void SwapInOutScheduler::wait_for_swap_out_first_iter() {
  while (used_bytes_swap_out > max_bytes_swap_out) {
    wait_for_swap_out_first_iter_impl();
  }
}


void SwapInOutScheduler::wait_for_all_swap_out_first_iter() {
  while (tail < order.size()) {
    wait_for_swap_out_first_iter_impl();
  }
}


void SwapInOutScheduler::wait_for_swap_out_first_iter_impl() {
  RecType& r = order[tail++];

  if (r.tag == RecTag::CLEAR) {
    return;
  }

  auto p = r.sawptr.lock();

  if (r.swapped_out) {
    // Wait for finish swapping out and release the source array of memory copy.
    if (p && p->head_array_class() == host_ctx.array_class &&
        p->get_num_arrays() > 0) { 
      // Not cleared yet, in first iteration, precleaer does be scheduled.
      p->get(p->dtype(), host_ctx, AsyncFlag::UNSAFE);
    }

    // Decrease memory usage
    r.swapped_out = false;
    used_bytes_swap_out -= r.swapped_out_bytes;
    r.swapped_out_bytes = 0;
  }
}


void  SwapInOutScheduler::swap_out_scheduled() {
  for (auto r : swap_out_schedule[func_idx]) {
    if (auto p = r.get().sawptr.lock()) {
      if (r.get().preclear) {
        p->clear();
        precleared[p] = true;
      }
      else if (!r.get().no_need_swap_out) {
        p->cast(p->dtype(), host_ctx, false, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
      }
    }
  }
}


void SwapInOutScheduler::wait_for_swap_out_scheduled() {
  for (auto r : wait_schedule[func_idx]) {
    wait_for_swap_out_scheduled_impl(r.get());
  }
}

void SwapInOutScheduler::wait_for_all_swap_out_scheduled() {
  for (auto r : wait_all_schedule) {
    wait_for_swap_out_scheduled_impl(r.get());
  }
}

  
void SwapInOutScheduler::wait_for_swap_out_scheduled_impl(const RecType& r) {
  if (r.no_need_swap_out) {
    return;
  }

  auto p = r.sawptr.lock();

  if (p && p->head_array_class() == host_ctx.array_class &&
      p->get_num_arrays() > 0) {
    p->get(p->dtype(), host_ctx, AsyncFlag::UNSAFE);
  }
}


void SwapInOutScheduler::swap_out_wrong_order() {
  for (int i = 0; i < wrong_ordered.size(); i++) {
    if (wrong_ordered[i].tag == RecTag::CLEAR) {
      continue;
    }

    if (wrong_ordered[i].ctx.array_class == device_ctx.array_class) {
      auto p = wrong_ordered[i].sawptr.lock();

      if (p && p->get_num_arrays() > 0) {
        // Not cleared yet. Swap out the array ***SYNCRONOUSLY***
        // because device synchronize will be called just after this call.
        p->cast(wrong_ordered[i].dtype, host_ctx, false);
      }
    }
    else if (wrong_ordered[i].ctx.array_class != host_ctx.array_class) {
      // Function used an array on an uncertain device
      NBLA_ERROR(error_code::type,
        "Unsupported array class: " + wrong_ordered[i].ctx.array_class);
    }
  }
}


//----------------------------------------------------------------
//  SyncedArrayCallback function
//----------------------------------------------------------------

void SwapInOutScheduler::set_synced_array_callback() {
  SingletonManager::
    get<SyncedArrayCallback>()->set_callback_func(synced_array_callback);
}

void SwapInOutScheduler::unset_synced_array_callback() {
  SingletonManager::get<SyncedArrayCallback>()->set_callback_func(nullptr);
}

void SwapInOutScheduler::
synced_array_callback_recorder(SyncedArrayPtr saptr,
                               const SyncedArrayCallbackTag func_name,
                               const dtypes dtype,
                               const Context &ctx,
                               const bool write_only) {
  auto tag = get_tag(func_name, write_only);

  if (synced_array_id_mapper.size() 
      > std::numeric_limits<unsigned int>::max()) {
    NBLA_ERROR(error_code::unclassified, 
               "Too many SyncedArray in excess of the max of unsigned int. " 
               "Please contact the developer to expand the size of SyncedArray ID.");
  }

  if (synced_array_id_mapper.find(saptr) == synced_array_id_mapper.end()) {
    auto next_id = static_cast<unsigned int>(synced_array_id_mapper.size());
    synced_array_id_mapper[saptr] = next_id;
  }

  order.push_back(RecType{tag, synced_array_id_mapper.at(saptr), saptr, 
                          saptr->size(), dtype, ctx, false, false, 0, false});
  synced_array_id_to_order_idx[synced_array_id_mapper.at(saptr)].push_back(order_idx);
  order_idx++;
}


void SwapInOutScheduler::
synced_array_callback_tracer(SyncedArrayPtr saptr,
                             const SyncedArrayCallbackTag func_name,
                             const dtypes dtype,
                             const Context &ctx,
                             const bool write_only) {
  auto tag = get_tag(func_name, write_only);

  // Return an error when encounting get/cast between preclear and actual clear.
  // It could happens due to the unpredicted change of the order.
  if (precleared[saptr]) {
    if (tag == RecTag::CLEAR) {
      precleared[saptr] = false;
    }
    else {
      NBLA_ERROR(error_code::target_specific_async, "Re-get/cast precleared array.");
    }
  }
  
  auto rec_saptr = order[order_idx].sawptr.lock();

  // Compare between the real and recorded order.
  if (!rec_saptr && // Expired
      order_idx < func_block_ends[func_idx] &&
      (tag == order[order_idx].tag &&
       saptr != rec_saptr &&
       saptr->size() == order[order_idx].size &&
       dtype == order[order_idx].dtype &&
       get_array_key_from_context(ctx) ==
       get_array_key_from_context(order[order_idx].ctx))) {
    // The SyncedArray is replaced in the current iteration.
    // Replace all recorded SyncedArray
    for (auto& i : synced_array_id_to_order_idx[order[order_idx].synced_array_id]) {
      order[i].sawptr = saptr;
    }
  }
  else if (order_idx >= func_block_ends[func_idx] ||
           (order_idx < func_block_ends[func_idx] && 
            (tag != order[order_idx].tag || 
             saptr != rec_saptr || 
             saptr->size() != order[order_idx].size ||
             dtype != order[order_idx].dtype ||
             get_array_key_from_context(ctx) != 
             get_array_key_from_context(order[order_idx].ctx)))) {
    // The number of real get/cast/clear is larger than that of the recorded
    // order, or the orders are different
    wrong_ordered.push_back({tag, 0, saptr, saptr->size(), dtype, ctx, 
                             false, false, 0, false});
  }

  order_idx++;
}


SwapInOutScheduler::RecTag
SwapInOutScheduler::get_tag(const SyncedArrayCallbackTag func_name,
  const bool write_only) {
  switch (func_name) {
  case SyncedArrayCallbackTag::GET:
    return RecTag::GETCAST;
    break;
  case SyncedArrayCallbackTag::CAST:
    return RecTag::GETCAST;
    break;
  case SyncedArrayCallbackTag::CLEAR:
    return RecTag::CLEAR;
    break;
  default:
    NBLA_ERROR(error_code::type, "Unsupported SyncedArrayCallbackTag");
    break;
  }
}
}
