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

#include <nbla/lms/swap_in_out_scheduler.hpp>
#include <nbla/host_stream_synchronizer_registry.hpp>
#include <nbla/singleton_manager.hpp>


namespace nbla {


SwapInOutScheduler::SwapInOutScheduler(const Context &h_ctx,
                                       const Context &d_ctx,
                                       const size_t bytes)
  : host_ctx(h_ctx), device_ctx(d_ctx),
    // Balancing the maximum GPU memory size for swap in/out in half
    max_bytes_swap_in(bytes / 2), max_bytes_swap_out(bytes / 2),
    // Set SyncedArrayCallback function for first iteration
    synced_array_callback([&](SyncedArrayPtr saptr, const SyncedArrayCallbackTag func_name,
                          const dtypes dtype, const Context &ctx, const bool write_only) {
               synced_array_callback_recorder(saptr, func_name, dtype, ctx, write_only); }) {}


SwapInOutScheduler::~SwapInOutScheduler() {}



//-----------------------------------------------------------------
//  Interfaces
//-----------------------------------------------------------------

// Start point of the scheduler
void SwapInOutScheduler::start_scheduling() {
  init();
  set_synced_array_callback();
}


// End point of scheduler
void SwapInOutScheduler::end_scheduling() {
  unset_synced_array_callback(); 
  finalize(); // This must be called after unset_synced_array_callback().
}


void SwapInOutScheduler::reset() {
  init();
  rec_order.clear();
  synced_array_callback = [&](SyncedArrayPtr saptr, const SyncedArrayCallbackTag func_name,
    const dtypes dtype, const Context &ctx, const bool write_only) {
    synced_array_callback_recorder(saptr, func_name, dtype, ctx, write_only); };
}


//----------------------------------------------------------------
//  Pre/post hook of function and update
//----------------------------------------------------------------

void SwapInOutScheduler::pre_function_callback(const CgFunctionPtr &ptr) { 
  pre_callback_impl(); 
}

void SwapInOutScheduler::post_function_callback(const CgFunctionPtr &ptr) {}

void SwapInOutScheduler::pre_update_callback() { 
  pre_callback_impl(); 
}

void SwapInOutScheduler::post_update_callback() {}



//----------------------------------------------------------------
//  Initialization
//----------------------------------------------------------------

// Initializer of the scheduler for each training iteration
void SwapInOutScheduler::init() {
  rec_head = 0;
  rec_tail = 0;
  used_bytes_swap_in = 0;
  used_bytes_swap_out = 0;
  swap_in_counts.clear();
  rec_use_idx = 0;
  function_idx = 0;
  wrong_ordered.clear();
  precleared.clear();
}


//----------------------------------------------------------------
//  Finalization
//----------------------------------------------------------------

// Finalizer of the scheduler for each training iteration
void SwapInOutScheduler::finalize() {
  // The post process of the last function of a network.
  if (function_idx > 0) {
    proc_for_prev_func();
  }

  // Wait to swap out all arrays before the next iteration.
  wait_swap_out(true);

  // Swap out all arrays out of the recorded order.
  // In the first iteration, wrong_ordered.size() is always zero.
  for (int i = 0; i < wrong_ordered.size(); i++) {
    if (wrong_ordered[i].tag != RecTag::CLEAR) {
      if (wrong_ordered[i].ctx.array_class == device_ctx.array_class) { // GPU array
        if (auto p = wrong_ordered[i].sawptr.lock()) { // weak pointer is alive
          if (p->get_num_arrays() > 0) { // The array not cleared yet
            // Swap out the array synchronously
            p->cast(wrong_ordered[i].dtype, host_ctx, false);
          }
        }
      }
      else if (wrong_ordered[i].ctx.array_class == host_ctx.array_class) { // CPU array
        // No need swap-out
      }
      else {
        // Function used an array on an uncertain device
        NBLA_ERROR(error_code::type,
          "Unsupported array class: " + wrong_ordered[i].ctx.array_class);
      }
    }
  }

  // Schedule the timings
  if (first_iter) {
    init();
    schedule();
  }
  
  /* Host should wait for the all asynchronous processes in GPU managed
  by the scheduler.

  If host does not wait, it is unsafe to modify the values of CPU memory.
  Example:
  1. An asynchronous data transfer (cudaMemcpyAsync) are
  writing or reading a CPU memory.
  2. Host process went out from the management of this scheduler.
  3. A user could change the values of that CPU memory used by cudaMemcpyAsync.
  4. Finally, data used by cudaMemcpyAsync was broken.

  In particular, one of the modification of CPU memory is the writing of
  the input data for the next training iteration.
  */
  HostStreamSynchronizer::synchronize(device_ctx);

  // After first itration, not record
  synced_array_callback = [&](SyncedArrayPtr saptr, const SyncedArrayCallbackTag func_name,
    const dtypes dtype, const Context &ctx, const bool write_only) {
    synced_array_callback_tracer(saptr, func_name, dtype, ctx, write_only); };

  first_iter = false;
}


//----------------------------------------------------------------
//  Scheduler
//----------------------------------------------------------------

vector<reference_wrapper<SwapInOutScheduler::RecType>>
SwapInOutScheduler::schedule_swap_in() {
  vector<reference_wrapper<SwapInOutScheduler::RecType>> scheduled_swap_in;

  while (rec_head < rec_order.size()) {
    RecType& r = rec_order[rec_head];

    if (r.tag == RecTag::CLEAR) {
      rec_head++;
    }
    else if (r.ctx.array_class == device_ctx.array_class) { // GPU array
      auto next_array_bytes = r.size * sizeof_dtype(r.dtype);

      // SyncedArray is not replaced
      // Fetch an GPU array
      if (used_bytes_swap_in + next_array_bytes > max_bytes_swap_in) {
        break; // Out of memory. Stop fetching.
      }
      else {
        if (swap_in_counts[r.synced_array_id][r.dtype] == 0) {
          if (!waiting_for_host_saptr[r.synced_array_id]) {
            // The array is firstly appeared in the queue.
            scheduled_swap_in.push_back(r);
          }
          // Increase memory usage
          used_bytes_swap_in += next_array_bytes;
        }

        // Increment the number of the same SyncedArray in the queue.
        swap_in_counts[r.synced_array_id][r.dtype]++;
      }

      rec_head++; // Move on the head of the queue
    }
    else if (r.ctx.array_class == host_ctx.array_class) { // CPU array
                                                          // No need swap-in (prefetch) to CPU. The array will be gotten/casted 
                                                          // synchronously by the function itself.
      rec_head++;

      // Stop prefetch until the end of use of the array.
      waiting_for_host_saptr[r.synced_array_id] = true;
    }
    else {
      // Get/cast of an array on an uncertain device
      NBLA_ERROR(error_code::type, "Unsupported array type: " + r.ctx.array_class);
    }
  }

  return scheduled_swap_in;
}


vector<reference_wrapper<SwapInOutScheduler::RecType>>
SwapInOutScheduler::schedule_swap_out(const int fid) {
  vector<reference_wrapper<SwapInOutScheduler::RecType>> scheduled_swap_out;

  for (size_t i = rec_function_ends[fid - 1]; i < rec_function_ends[fid]; i++) {
    RecType& r = rec_order[i];

    if (r.tag != RecTag::CLEAR) {
      if (r.ctx.array_class == device_ctx.array_class) { // GPU array
          // SyncedArray is not replaced.
        if (accumulate_counts(swap_in_counts[r.synced_array_id]) == 1) {
          // In the first iteration, arrays used in a function are 
          // always swapped out. After the first itreation, an array
          // are swapped out when the same array is no longer in the queue.
          // Swap out the array
          scheduled_swap_out.push_back(r);

          auto array_bytes = r.size * sizeof_dtype(r.dtype);
          used_bytes_swap_out += array_bytes; // Increase memory usage
          r.swapped_out = true;
          r.swapped_out_bytes = array_bytes;
        }

        // Decrease memory usage after the final use of the prefetched array 
        if (swap_in_counts[r.synced_array_id][r.dtype] == 1) {
          auto array_bytes = r.size * sizeof_dtype(r.dtype);
          used_bytes_swap_in -= array_bytes;
        }

        // Decrease the counts of a used array in the queue.
        swap_in_counts[r.synced_array_id][r.dtype]--;
      }
      else if (r.ctx.array_class == host_ctx.array_class) { // CPU array
                                                                       // No need swap-out
      }
      else {
        // Get/cast of an array on an uncertain device
        NBLA_ERROR(error_code::type, "Unsupported array type: " + r.ctx.array_class);
      }
    }
  }

  return scheduled_swap_out;
}


vector<reference_wrapper<SwapInOutScheduler::RecType>>
SwapInOutScheduler::schedule_wait_for_swap_out() {
  vector<reference_wrapper<SwapInOutScheduler::RecType>> scheduled_wait;

  // When out of memory, wait to finish swap-out and release memory.
  while (used_bytes_swap_out > max_bytes_swap_out && rec_tail < rec_order.size()) {
    RecType& r = rec_order[rec_tail++];

    if (r.tag != RecTag::CLEAR) {
      if (r.swapped_out) {
        // SyncedArray is not replaced.
        // Wait to finish swap-out and release the source array of memory copy.
        scheduled_wait.push_back(r);

        // Decrease memory usage
        used_bytes_swap_out -= r.swapped_out_bytes;
        r.swapped_out = false;
        r.swapped_out_bytes = 0;
      }
    }
  }

  if (used_bytes_swap_out > max_bytes_swap_out) {
    NBLA_ERROR(error_code::memory,
      "Out of GPU memory for swap-out");
  }

  return scheduled_wait;
}


/* Determine preclear timing

For the same SyncedArray, get/cast just before the clear is
time to preclear it instead of swap out it.
*/
void SwapInOutScheduler::schedule_preclear() {
  unordered_map<unsigned int, bool> clear_flag;

  for (int i = rec_order.size() - 1; i >= 0; i--) {
    if (rec_order[i].tag == RecTag::CLEAR) {
      clear_flag[rec_order[i].synced_array_id] = true;
    }
    else {
      rec_order[i].preclear = clear_flag[rec_order[i].synced_array_id];
      clear_flag[rec_order[i].synced_array_id] = false;
    }
  }
}


void SwapInOutScheduler::schedule() {
  auto last_function = rec_function_ends.size() - 1;

  // Virtually iterate all layer functions and solver update
  for (int fid = 0; fid < last_function; fid++) {
    prefetch_schedule[fid] = schedule_swap_in();

    if (rec_head < rec_function_ends[fid]) {
      NBLA_ERROR(error_code::memory,
        "Some arrays were not prefetched probably due to out of GPU memory. ");
    }

    swap_out_schedule[fid] = schedule_swap_out(fid);
    wait_schedule[fid] = schedule_wait_for_swap_out();
  }

  schedule_preclear();
}



//----------------------------------------------------------------
//  Execution
//----------------------------------------------------------------

/** Common implementation of pre callback
*/
void SwapInOutScheduler::pre_callback_impl() {
  unset_synced_array_callback(); // Avoid unnecessary record and trace

  if (function_idx > 0) {
    proc_for_prev_func(); // post process of the previous function
  }
  proc_for_next_func(); // pre process of the next function
  
  set_synced_array_callback(); // Restart record or trace
}

/** The post-process of the previous function.
*/
void SwapInOutScheduler::proc_for_prev_func() {
  // Record the end of a function.
  if (first_iter) {
    rec_function_ends.push_back(rec_use_idx);
  }

  // Swap out and preclear the arrays used in the previous function.
  swap_out();

  if (rec_use_idx < rec_function_ends[function_idx]) {
    /* If the number of get/cast/clear in this iteration is less than
       recorded one, reset the index the recorded start of the next function 
       in order to compare between recorded and actually used order in this iteration. 

       See synced_array_callback_tracer about the comparison.
    */
    rec_use_idx = rec_function_ends[function_idx];
  }
}

/* The pre-process of the next function.
*/
void SwapInOutScheduler::proc_for_next_func() {
  function_idx++;
  if (function_idx > rec_function_ends.size()) {
    NBLA_ERROR(error_code::unclassified, "Different the number of functions");
  }

  if (!first_iter) {
    swap_in(); // Prefetch arrays as possible.
  }
}

/* Swap in (prefetch)
*/
void SwapInOutScheduler::swap_in() {
  auto scheduled_prefetch = prefetch_schedule[function_idx - 1];

  for (auto r : scheduled_prefetch) {
    if (auto p = r.get().sawptr.lock()) {
      p->get(r.get().dtype, r.get().ctx, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
    }
  }
}

/** Swap out
*/
void SwapInOutScheduler::swap_out() {
  // Swap out the arrays used in the previous function
  swap_out_impl();

  // When out of memory, wait to finish swap-out and release memory.
  wait_swap_out();
 
  if (used_bytes_swap_out > max_bytes_swap_out) {
    NBLA_ERROR(error_code::memory,
               "Out of GPU memory for swap-out");
  }
}

/** The implementation of swap out
*/
void SwapInOutScheduler::swap_out_impl() {
  if (first_iter) {
    for (size_t i = rec_function_ends[function_idx - 1];
      i < rec_function_ends[function_idx]; i++) {
      if (rec_order[i].tag != RecTag::CLEAR) {
        if (rec_order[i].ctx.array_class == device_ctx.array_class) { // GPU array
          if (auto p = rec_order[i].sawptr.lock()) {
            // SyncedArray is not replaced.
            // In the first iteration, arrays used in a function are 
            // always swapped out.
            if (p->get_num_arrays() > 0) {
              // The array is not cleared yet.
              // Swap out the array
              p->cast(p->dtype(), host_ctx, false, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
              auto array_bytes = p->size() * sizeof_dtype(p->dtype());
              used_bytes_swap_out += array_bytes; // Increase memory usage
              rec_order[i].swapped_out = true;
              rec_order[i].swapped_out_bytes = array_bytes;
            }
          }
        }
        else if (rec_order[i].ctx.array_class == host_ctx.array_class) { // CPU array
          // No need swap-out
        }
        else {
          // Get/cast of an array on an uncertain device
          NBLA_ERROR(error_code::type, "Unsupported array type: " + rec_order[i].ctx.array_class);
        }
      }
    }
  }
  else {
    auto scheduled_swap_out = swap_out_schedule[function_idx - 1];

    for (auto r : scheduled_swap_out) {
      if (auto p = r.get().sawptr.lock()) {
        if (r.get().preclear) {
          p->clear();
          precleared[p] = true;
        }
        else {
          p->cast(p->dtype(), host_ctx, false, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
        }
      }
    }
  }
}

/* Wait to finish swap-out and release memory.
*/
void SwapInOutScheduler::wait_swap_out(const bool final) {

  if (first_iter || final) {
    while ((final || used_bytes_swap_out > max_bytes_swap_out) && rec_tail < rec_order.size()) {
      RecType& r = rec_order[rec_tail++];

      if (r.tag != RecTag::CLEAR) {
        if (r.swapped_out) {
          if (auto p = r.sawptr.lock()) {
            // SyncedArray is not replaced.
            // Wait to finish swap-out and release the source array of memory copy.
            if (p->head_array_class() == host_ctx.array_class && p->get_num_arrays() > 0) {
              // Swap out the array
              p->get(p->dtype(), host_ctx, AsyncFlag::UNSAFE);
            }

            // Decrease memory usage
            used_bytes_swap_out -= r.swapped_out_bytes;
            r.swapped_out = false;
            r.swapped_out_bytes = 0;
          }
        }
      }
    }
  }
  else {
    auto scheduled_wait = wait_schedule[function_idx - 1];

    for (auto r : scheduled_wait) {
      if (auto p = r.get().sawptr.lock()) {
        if (p->head_array_class() == host_ctx.array_class && p->get_num_arrays() > 0) {
          p->get(p->dtype(), host_ctx, AsyncFlag::UNSAFE);
        }
      }
    }
  }
}



//----------------------------------------------------------------
//  SyncedArrayCallback function
//----------------------------------------------------------------

// Setter of SyncedArray callback
void SwapInOutScheduler::set_synced_array_callback() {
  SingletonManager::get<SyncedArrayCallback>()->set_callback_func(synced_array_callback);
}

// Remover of SyncedArray callback
void SwapInOutScheduler::unset_synced_array_callback() {
  SingletonManager::get<SyncedArrayCallback>()->set_callback_func(nullptr);
}

/// SyncedArray callback function to record get/cast/clear in the first iteration.
void SwapInOutScheduler::
synced_array_callback_recorder(SyncedArrayPtr saptr,
                               const SyncedArrayCallbackTag func_name,
                               const dtypes dtype,
                               const Context &ctx,
                               const bool write_only) {
  auto tag = get_tag(func_name, write_only);

  if (synced_array_id_mapper.size() > UINT_MAX) {
    NBLA_ERROR(error_code::unclassified, 
               "Too many SyncedArray in excess of the max of unsigned int. " 
               "Please contact the developer to expand the size of SyncedArray ID.");
  }

  if (synced_array_id_mapper.find(saptr) == synced_array_id_mapper.end()) {
    auto next_id = static_cast<unsigned int>(synced_array_id_mapper.size());
    synced_array_id_mapper[saptr] = next_id;
  }

  rec_order.push_back(RecType{tag, synced_array_id_mapper.at(saptr), saptr, 
                              saptr->size(), dtype, ctx, false, false, 0});
  rec_use_idx++;
}


// SyncedArray callback function to trace get/cast/clear after the first iteration.
void SwapInOutScheduler::
synced_array_callback_tracer(SyncedArrayPtr saptr,
                             const SyncedArrayCallbackTag func_name,
                             const dtypes dtype,
                             const Context &ctx,
                             const bool write_only) {
  auto tag = get_tag(func_name, write_only);

  // Return an error when encounting get/cast between preclear and actual clear.
  // It could happens due to unpredicted change from the recorded order.
  if (precleared[saptr]) {
    if (tag == RecTag::CLEAR) {
      precleared[saptr] = false;
    }
    else {
      NBLA_ERROR(error_code::target_specific_async, "Re-get/cast precleared array.");
    }
  }

  auto rec_saptr = rec_order[rec_use_idx].sawptr.lock();

  // Compare between the real and recorded order.
  if (rec_use_idx < rec_function_ends[function_idx] &&
      (tag == rec_order[rec_use_idx].tag ||
       saptr != rec_saptr ||
       dtype == rec_order[rec_use_idx].dtype ||
       get_array_key_from_context(ctx) ==
       get_array_key_from_context(rec_order[rec_use_idx].ctx))) {
    // The SyncedArray is replaced in the current iteration.
    rec_order[rec_use_idx].sawptr = saptr;
  }
  else if (rec_use_idx >= rec_function_ends[function_idx] ||
           (rec_use_idx < rec_function_ends[function_idx] && 
            (tag != rec_order[rec_use_idx].tag || 
             saptr != rec_saptr || 
             dtype != rec_order[rec_use_idx].dtype ||
             get_array_key_from_context(ctx) != 
             get_array_key_from_context(rec_order[rec_use_idx].ctx)))) {
    // The number of real get/cast/clear is larger than that of the recorded order,
    // or the orders are different
    wrong_ordered.push_back({tag, 0, saptr, saptr->size(), dtype, ctx, false, false, 0});
  }

  rec_use_idx++;
}


// Conversion of the recorded tag
SwapInOutScheduler::RecTag
SwapInOutScheduler::get_tag(const SyncedArrayCallbackTag func_name,
  const bool write_only) {
  switch (func_name) {
  case SyncedArrayCallbackTag::GET:
    return RecTag::GET;
    break;
  case SyncedArrayCallbackTag::CAST:
    if (write_only) return RecTag::CAST_WRITE_ONLY;
    else return RecTag::CAST;
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