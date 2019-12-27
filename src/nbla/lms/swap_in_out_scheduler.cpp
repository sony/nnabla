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

/* Constructor
 */
SwapInOutScheduler::SwapInOutScheduler(const Context &h_ctx,
                                       const Context &d_ctx,
                                       const size_t bytes)
  : host_ctx(h_ctx), 
    device_ctx(d_ctx),
    max_bytes_swap_in(bytes),
    max_bytes_swap_out(bytes / 2), // Balance the max GPU memory in half
    synced_array_callback([&](SyncedArrayPtr saptr, 
                              const SyncedArrayCallbackTag sa_tag,
                              const dtypes dtype,
                              const Context &ctx,
                              const bool write_only) {
      // Set SyncedArrayCallback function for first iteration
      synced_array_callback_recorder(saptr, sa_tag, dtype, 
                                     ctx, write_only);}) 
{
    // Create non blocking streams for data transfer
    BackendUtils::create_lms_streams(d_ctx);
}


/* Destructor
 */
SwapInOutScheduler::~SwapInOutScheduler() {}


/* User Interfaces to start the scheduling code block
 */
void SwapInOutScheduler::start_scheduling() {
  // Initialize
  order_idx = 0;
  func_idx = 0;
  wrong_ordered.clear();
  precleared.clear();

  // Clear variables used in the first iteration
  said_map.clear();

  // Set the SyncedArrayCallback
  set_synced_array_callback();
}


/* User Interfaces to end the scheduling code block
 */
void SwapInOutScheduler::end_scheduling() {
  // Unset the SyncedArrayCallback
  unset_synced_array_callback();
  
  // Post process of the last function.
  swap_out_step();

  if (first_iter) {
    // Wait for swaping out all arrays and clean GPU memory for the next iteration.
    wait_for_all_swap_out_first_iter();

    if (used_bytes_swap_out_first_iter != 0) {
      NBLA_ERROR(error_code::unclassified, "used_bytes_swap_out != 0");
    }
    if (tail_first_iter != order.size()) {
      NBLA_ERROR(error_code::unclassified, "tail != order.size()");
    }

    // Schedule at the end of first iteration
    schedule();
  }
  else {
    // Wait for swaping out all arrays and clean GPU memory for the next iteration.
    wait_for_all_swap_out_scheduled();

    // Swap out all disordered arrays
    swap_out_wrong_order();
  }

  /* Host must wait for the all asynchronous memory manipulation for safety.
     That is because data on CPU memory can be destroyed, for esample,
     by the conflict between
     1. asynchronous swap out
     2. the loads of input data and label for the next iteration.
  */
  BackendUtils::device_synchronize(device_ctx);
  
  // After first itration, synced array callback is replaced.
  synced_array_callback = [&](SyncedArrayPtr saptr, 
                              const SyncedArrayCallbackTag sa_tag,
                              const dtypes dtype, 
                              const Context &ctx,
                              const bool write_only) {
    synced_array_callback_tracer(saptr, sa_tag, dtype, ctx, write_only);};

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
void SwapInOutScheduler::schedule() {
  /* This counts the number of same arrays in the queue.
     If count of an array > 0, no need to fetch the same array again.
     If count of an array > 1, no need to swap out the array because it is
     planed to be used in the queue.
  */
  SyncedArrayCounts synced_array_counts;

  int head = 0;
  int tail = 0;
  size_t used_bytes_swap_in = 0;
  size_t used_bytes_swap_out = 0;
  auto last_function = func_block_ends.size();
  unordered_map<unsigned int, bool> host_uses_this_synced_array;

  // They are used to remove uneccesary swap-out
  unordered_map<unsigned int, bool> swapped_out;
  unordered_map<unsigned int, RecType*> swapped_out_r;

  // Preclear schedule will be used in swap-out schedule.
  schedule_preclear();

  // Virtually iterate all layer functions and solver update
  int fid = 0;

  // Calculate used GPU memory before forward starts,
  // and swap out if necessary.
  detect_swap_in_before_forward(head, used_bytes_swap_in,
                                synced_array_counts);
  swap_out_schedule[fid] = schedule_swap_out(fid, 
                                             used_bytes_swap_in,
                                             used_bytes_swap_out,
                                             synced_array_counts,
                                             swapped_out, swapped_out_r);
  wait_schedule[fid] = schedule_wait_for_swap_out(tail, used_bytes_swap_out,
                                                  swapped_out, swapped_out_r);

  // Forward, backward, update
  for (fid = 1; fid < last_function; fid++) {
    swap_in_schedule[fid] = schedule_swap_in(head, fid, 
                                             used_bytes_swap_in,
                                             used_bytes_swap_out,
                                             synced_array_counts,
                                             host_uses_this_synced_array,
                                             swapped_out, swapped_out_r);

    if (head < func_block_ends[fid]) {
      NBLA_ERROR(error_code::memory,
        "Some arrays were not prefetched probably due to out of GPU memory. ");
    }

    swap_out_schedule[fid] = schedule_swap_out(fid,
                                               used_bytes_swap_in,
                                               used_bytes_swap_out,
                                               synced_array_counts,
                                               swapped_out, swapped_out_r);

    wait_schedule[fid] = schedule_wait_for_swap_out(tail, used_bytes_swap_out, 
                                                    swapped_out, swapped_out_r);
  }

  wait_all_schedule = schedule_wait_for_all_swap_out(tail, used_bytes_swap_out, 
                                                     swapped_out, swapped_out_r);

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
      RecType *r = &order[i];

      if (r->tag == RecTag::CLEAR) {
        continue;
      }

      if (r->ctx.array_class == host_ctx.array_class) {
        host_func = true;
      }
    }

    is_host_func.push_back(host_func);
  }
}


void SwapInOutScheduler::
detect_swap_in_before_forward(int& head, size_t& used_bytes_swap_in,
                              SyncedArrayCounts& synced_array_counts) {
  while (head < func_block_ends[0]) {
    RecType *r = &order[head];

    if (r->tag == RecTag::CLEAR) {
      head++;
      continue;
    }

    if (r->ctx.array_class == device_ctx.array_class) {
      // All fetches were already done. Just calculate memory size.
      auto array_bytes = r->size * sizeof_dtype(r->dtype);

      // First fetch
      if (synced_array_counts[r->said][r->dtype] == 0) {
        used_bytes_swap_in += array_bytes;
      }

      // Increment the number of the same SyncedArray in the queue.
      synced_array_counts[r->said][r->dtype]++;
      head++; // Move on the head of the queue
    }
    else if (r->ctx.array_class == host_ctx.array_class) {
      // Because func_idx == 0 means all get/cast finished already
      // the host process must be finished.

      head++; // Move on the head of the queue
    }
    else {
      // Get/cast of an array on an uncertain device
      NBLA_ERROR(error_code::type,
                 "Unsupported array type: " + r->ctx.array_class);
    }
  }
}


vector<SwapInOutScheduler::RecType*>
SwapInOutScheduler::
schedule_swap_in(int& head, const int fid, 
                 size_t& used_bytes_swap_in, size_t& used_bytes_swap_out,
                 SyncedArrayCounts& synced_array_counts,
                 unordered_map<unsigned int, bool>& host_uses_this_synced_array,
                 unordered_map<unsigned int, bool>& swapped_out,
                 unordered_map<unsigned int, RecType*>& swapped_out_r) {
  vector<RecType*> schedule;

  while (head < order.size()) {
    RecType *r = &order[head];

    if (r->tag == RecTag::CLEAR) {
      head++;
      continue;
    }
    
    if (r->ctx.array_class == device_ctx.array_class) {
      auto next_array_bytes = r->size * sizeof_dtype(r->dtype);

      if (used_bytes_swap_in + next_array_bytes
          > max_bytes_swap_in - used_bytes_swap_out) {
        break; // Out of memory. Stop fetching.
      }

      // Fetch
      if (synced_array_counts[r->said][r->dtype] == 0) {
        // The array is firstly appeared in the queue.

        if (!host_uses_this_synced_array[r->said]) {
          schedule.push_back(r);

          // If the array was previously swapped out,
          // the memcpy was waited for by swap in.
          if (swapped_out[r->said]) {
            auto rptr = swapped_out_r[r->said];
            
            // The array is used before swap out is completed.
            // It is not required to swap out the array.
            rptr->no_need_swap_out = true;

            // Remove memory size from swap-out memory
            rptr->swapped_out = false;
            used_bytes_swap_out -= rptr->swapped_out_bytes;
            rptr->swapped_out_bytes = 0;

            // reset flag
            swapped_out[r->said] = false;
            swapped_out_r[r->said] = nullptr;
          }
        }

        // Increase memory usage
        used_bytes_swap_in += next_array_bytes;
      }

      // Increment the number of the same SyncedArray in the queue.
      synced_array_counts[r->said][r->dtype]++;
      head++; // Move on the head of the queue
    }
    else if (r->ctx.array_class == host_ctx.array_class) {
      // No need swap-in (prefetch) to CPU. The array will be gotten/casted 
      // synchronously by the function itself. 
      // Stop prefetch these type of arrays.
      if (fid > 0) {
        // Because func_idx == 0 means all get/cast finished already
        // the host process must be finished.
        host_uses_this_synced_array[r->said] = true;
      }

      // The arrray comes on to host by swap out.
      if (swapped_out[r->said]) {
        // reset flag
        swapped_out[r->said] = false;
        swapped_out_r[r->said] = nullptr;
      }

      head++;
    }
    else {
      // Get/cast of an array on an uncertain device
      NBLA_ERROR(error_code::type, 
                 "Unsupported array type: " + r->ctx.array_class);
    }
  }

  return schedule;
}


vector<SwapInOutScheduler::RecType*>
SwapInOutScheduler::
schedule_swap_out(const int fid,
                  size_t& used_bytes_swap_in, size_t& used_bytes_swap_out,
                  SyncedArrayCounts& synced_array_counts, 
                  unordered_map<unsigned int, bool>& swapped_out,
                  unordered_map<unsigned int, RecType*>& swapped_out_r) {
  vector<RecType*> schedule;

  for (size_t i = (fid == 0 ? 0 : func_block_ends[fid - 1]);
              i < func_block_ends[fid];
              i++) {
    RecType *r = &order[i];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }

    if (r->ctx.array_class == device_ctx.array_class) { 
      if (accumulate_counts(synced_array_counts[r->said]) == 1) {
        // An array is swapped out when the same array is no longer
        // in the queue.
        if (std::find(preclear_schedule[fid].begin(), 
                      preclear_schedule[fid].end(), r)
            == preclear_schedule[fid].end()) {
          // Not precleared
          schedule.push_back(r);

          r->swapped_out = true;
          swapped_out[r->said] = true;
          swapped_out_r[r->said] = &order[i];
        
          // Transfer memory usage of all types
          r->swapped_out_bytes = 0;

          for (auto it : synced_array_counts[r->said]) {
            auto array_bytes = r->size * sizeof_dtype(it.first);
            used_bytes_swap_out += array_bytes;
            r->swapped_out_bytes += array_bytes;
          }
        }

        // Transfer memory usage of all types
        for (auto it : synced_array_counts[r->said]) {
          auto array_bytes = r->size * sizeof_dtype(it.first);
          used_bytes_swap_in -= array_bytes;
        }

        // Reset usage
        synced_array_counts[r->said].clear();
      }
      else {
        // Decrease the counts of a used array in the queue.
        synced_array_counts[r->said][r->dtype]--;
      }
    }
    else if (r->ctx.array_class != host_ctx.array_class) {
      // Get/cast of an array on an uncertain device
      NBLA_ERROR(error_code::type, 
                 "Unsupported array type: " + r->ctx.array_class);
    }
  }

  return schedule;
}


vector<SwapInOutScheduler::RecType*>
SwapInOutScheduler::schedule_wait_for_swap_out(
  int& tail, size_t& used_bytes_swap_out,
  unordered_map<unsigned int, bool>& swapped_out,
  unordered_map<unsigned int, RecType*>& swapped_out_r) 
{
  vector<RecType*> schedule;

  // When out of memory, wait to finish swap-out and release memory.
  while (used_bytes_swap_out > max_bytes_swap_out) {
    schedule_wait_for_swap_out_impl(schedule, tail, used_bytes_swap_out, 
                                    swapped_out, swapped_out_r);
  }

  return schedule;
}


vector<SwapInOutScheduler::RecType*>
SwapInOutScheduler::schedule_wait_for_all_swap_out(
  int& tail, size_t& used_bytes_swap_out, 
  unordered_map<unsigned int, bool>& swapped_out,
  unordered_map<unsigned int, RecType*>& swapped_out_r)
{
  vector<RecType*> schedule;

  // When out of memory, wait for finishing swap out.
  while (tail < order.size()) {
    schedule_wait_for_swap_out_impl(schedule, tail, used_bytes_swap_out,
                                    swapped_out, swapped_out_r);
  }

  return schedule;
}


void SwapInOutScheduler::schedule_wait_for_swap_out_impl(
  vector<SwapInOutScheduler::RecType*>& schedule,
  int& tail, size_t& used_bytes_swap_out,
  unordered_map<unsigned int, bool>& swapped_out,
  unordered_map<unsigned int, RecType*>& swapped_out_r)
{
  RecType *r = &order[tail++];

  if (r->swapped_out) {
    // Wait for finishing swap out and release the source array of memory copy.
    schedule.push_back(r);

    // Decrease memory usage
    r->swapped_out = false;
    used_bytes_swap_out -= r->swapped_out_bytes;
    r->swapped_out_bytes = 0;

    swapped_out[r->said] = false;
    swapped_out_r[r->said] = nullptr;
  }
}


// For the same SyncedArray, get/cast just before the clear is
// time to preclear it instead of swapping it out.
void SwapInOutScheduler::schedule_preclear() {
  unordered_map<unsigned int, bool> clear_flag;
  int fid = func_block_ends.size() - 1;

  for (int i = order.size() - 1; i >= 0; i--) {
    RecType *r = &order[i];

    if (i < func_block_ends[fid - 1]) {
      fid--;
    }

    if (fid == 0) {
      // fid == 0 is before the first pre-function hook.
      // No chance to preclear.
      break;
    }

    if (r->tag == RecTag::CLEAR) {
      clear_flag[r->said] = true;
    }
    else if (clear_flag[r->said]) {
      preclear_schedule[fid].push_back(r);
      clear_flag[r->said] = false;
    }
  }
}



//----------------------------------------------------------------
//  Execute swap in/out
//----------------------------------------------------------------
// Common implementation of pre callback
void SwapInOutScheduler::pre_callback() {
  unset_synced_array_callback(); // Avoid unnecessary record and trace

  // Post process of the previous function 
  swap_out_step(); 

  // Pre process of the next function
  swap_in_step();
  
  if (!first_iter && is_host_func[func_idx]) {
    BackendUtils::default_stream_synchronize(device_ctx);
  }

  set_synced_array_callback(); // Restart record or trace
}


// Common implementation of post callback
void SwapInOutScheduler::post_callback() {
  if (!first_iter) {
    unset_synced_array_callback(); // Avoid unnecessary record and trace
    preclear_scheduled(); // preclear
    set_synced_array_callback(); // Restart record or trace
  }
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
    if (auto p = r->sawptr.lock()) {
      p->get(r->dtype, device_ctx, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
    }
  }
}

// Swap out
void SwapInOutScheduler::swap_out() {
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
  SyncedArrayCounts synced_array_counts;
  const int start_idx = func_idx == 0 ? 0 : func_block_ends[func_idx - 1];

  for (int i = start_idx; i < func_block_ends[func_idx]; i++) {
    RecType *r = &order[i];
    if (r->tag == RecTag::CLEAR) continue;

    if (r->ctx.array_class == device_ctx.array_class) {
      synced_array_counts[r->said][r->dtype]++;
    }
    else if (r->ctx.array_class != host_ctx.array_class) {
      // Get/cast of an array on an uncertain device
      NBLA_ERROR(error_code::type, "Unsupported array type: " + r->ctx.array_class);
    }
  }

  // Swap out
  for (int i = start_idx; i < func_block_ends[func_idx]; i++) {
    RecType *r = &order[i];
    if (r->tag == RecTag::CLEAR) continue;

    if (r->ctx.array_class == device_ctx.array_class) {
      if (accumulate_counts(synced_array_counts[r->said]) == 1) {
        auto p = r->sawptr.lock();

        if (p && is_not_cleared_yet(p)) {
          // The array is not cleared yet. Swap out the array
          p->cast(p->dtype(), host_ctx, false, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
          r->swapped_out = true;

          // Counts memory usage of all types
          r->swapped_out_bytes = 0;

          for (auto it : synced_array_counts[r->said]) {
            auto array_bytes = r->size * sizeof_dtype(it.first);
            used_bytes_swap_out_first_iter += array_bytes;
            r->swapped_out_bytes += array_bytes;
          }
        }

        synced_array_counts[r->said].clear();
      }
      else {
        synced_array_counts[r->said][r->dtype]--;
      }
    }
  }
}

void SwapInOutScheduler::wait_for_swap_out_first_iter() {
  while (used_bytes_swap_out_first_iter > max_bytes_swap_out) {
    wait_for_swap_out_first_iter_impl();
  }
}


void SwapInOutScheduler::wait_for_all_swap_out_first_iter() {
  while (tail_first_iter < order.size()) {
    wait_for_swap_out_first_iter_impl();
  }
}


void SwapInOutScheduler::wait_for_swap_out_first_iter_impl() {
  RecType *r = &order[tail_first_iter++];

  if (r->tag == RecTag::CLEAR) {
    return;
  }

  auto p = r->sawptr.lock();

  if (r->swapped_out) {
    // Wait for finish swapping out and release the source array of memory copy.
    if (p && p->head_array_class() == host_ctx.array_class &&
        is_not_cleared_yet(p)) { 
      // Not cleared yet, in first iteration, precleaer does be scheduled.
      p->get(p->dtype(), host_ctx, AsyncFlag::UNSAFE);
    }

    // Decrease memory usage
    r->swapped_out = false;
    used_bytes_swap_out_first_iter -= r->swapped_out_bytes;
    r->swapped_out_bytes = 0;
  }
}


void SwapInOutScheduler::preclear_scheduled() {
  for (auto r : preclear_schedule[func_idx]) {
    if (auto p = r->sawptr.lock()) {
      p->clear();
      precleared[p] = true;
    }
  }
}


void SwapInOutScheduler::swap_out_scheduled() {
  for (auto r : swap_out_schedule[func_idx]) {
    if (auto p = r->sawptr.lock()) {
      if (!r->no_need_swap_out) {
        p->cast(p->dtype(), host_ctx, false, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
      }
    }
  }
}


void SwapInOutScheduler::wait_for_swap_out_scheduled() {
  for (auto r : wait_schedule[func_idx]) {
    wait_for_swap_out_scheduled_impl(r);
  }
}

void SwapInOutScheduler::wait_for_all_swap_out_scheduled() {
  for (auto r : wait_all_schedule) {
    wait_for_swap_out_scheduled_impl(r);
  }
}

  
void SwapInOutScheduler::wait_for_swap_out_scheduled_impl(const RecType *r) {
  if (r->no_need_swap_out) {
    return;
  }

  auto p = r->sawptr.lock();

  if (p && p->head_array_class() == host_ctx.array_class &&
      is_not_cleared_yet(p)) {
    p->get(p->dtype(), host_ctx, AsyncFlag::UNSAFE);
  }
}


void SwapInOutScheduler::swap_out_wrong_order() {
  for (int i = 0; i < wrong_ordered.size(); i++) {
    RecType *r = &wrong_ordered[i];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }

    if (r->ctx.array_class == device_ctx.array_class) {
      auto p = r->sawptr.lock();

      if (p && is_not_cleared_yet(p)) {
        // Swap out the array SYNCRONOUSLY because device synchronize will be 
        // called just after this.
        p->cast(r->dtype, host_ctx, false);
      }
    }
    else if (r->ctx.array_class != host_ctx.array_class) {
      // Function used an array on an uncertain device
      NBLA_ERROR(error_code::type,
                 "Unsupported array class: " + r->ctx.array_class);
    }
  }
}


//----------------------------------------------------------------
//  SyncedArrayCallback function
//----------------------------------------------------------------
/* Setter
 */
void SwapInOutScheduler::set_synced_array_callback() {
  SingletonManager::get<SyncedArrayCallback>()
    ->set_callback_func(synced_array_callback);
}

/* Unsetter
 */
void SwapInOutScheduler::unset_synced_array_callback() {
  SingletonManager::get<SyncedArrayCallback>()
    ->set_callback_func(nullptr);
}

/* SyncedArrayCallback function to record the order
 */
void SwapInOutScheduler::
synced_array_callback_recorder(SyncedArrayPtr saptr,
                               const SyncedArrayCallbackTag sa_tag,
                               const dtypes dtype,
                               const Context &ctx,
                               const bool write_only) {
  auto tag = convert_tag(sa_tag, write_only);

  // Define SyncedArray ID
  if (said_map.find(saptr) == said_map.end()) {
    said_map[saptr] = static_cast<unsigned int>(said_map.size());
  }
  auto said = said_map.at(saptr);

  // Record the order
  order.push_back(RecType{tag, said, saptr, saptr->size(), dtype, ctx});
  said_to_order_idx[said].push_back(order_idx);
  order_idx++;
}


/* SyncedArrayCallback function to trace the recorded order
 */
void SwapInOutScheduler::
synced_array_callback_tracer(SyncedArrayPtr saptr,
                             const SyncedArrayCallbackTag sa_tag,
                             const dtypes dtype,
                             const Context &ctx,
                             const bool write_only) {
  auto tag = convert_tag(sa_tag, write_only);

  // If unexpected get or cast appears between preclear and actual clear,
  // the deleted data will be used and then the whole computation will be
  // destroyed. In this case, abort.
  if (precleared[saptr]) {
    // This SyncedArray was cleared
    if (tag == RecTag::CLEAR) {
      // Actual clear. it is Ok.
      precleared[saptr] = false;
    }
    else {
      // Get or cast. Abort.
      NBLA_ERROR(error_code::unclassified,
                 "Unexpected get or cast appers after preclear.");
    }
  }

  // Check if the on-going iteration follows the recorded order.
  auto r = order[order_idx];
  auto rec_saptr = r.sawptr.lock();

  // Case: Over the current function block.
  if (order_idx >= func_block_ends[func_idx]) {
    wrong_ordered.push_back({tag, 0, saptr, saptr->size(), dtype, ctx});
  }
  // Case: SyncedArray replacement in the same order
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
  // Case: Different entry
  else if (tag != r.tag || 
           saptr != rec_saptr ||
           saptr->size() != r.size ||
           dtype != r.dtype ||
           ctx.array_class != r.ctx.array_class) {
   wrong_ordered.push_back({tag, 0, saptr, saptr->size(), dtype, ctx});
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
