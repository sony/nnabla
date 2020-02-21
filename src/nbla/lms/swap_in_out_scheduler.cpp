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
                                       const size_t bytes,
                                       const size_t prefetch_bytes)
  : host_ctx(h_ctx), 
    device_ctx(d_ctx),
    max_bytes(bytes),
    max_prefetch_bytes(prefetch_bytes),
    synced_array_callback([&](SyncedArrayPtr saptr, 
                              const SyncedArrayCallbackTag sa_tag,
                              const dtypes dtype,
                              const Context &ctx,
                              const bool write_only,
                              const bool first_creation) {
      // Set SyncedArrayCallback function for first iteration
      synced_array_callback_recorder(saptr, sa_tag, dtype, 
                                     ctx, write_only, first_creation);}) 
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
  // Swap out and preclear the arrays used in the previous function.
  if (first_iter) {
    // Record the end of a function.
    func_block_ends.push_back(order_idx);
    swap_out_first_iter();
    wait_for_swap_out_first_iter();


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
    if (order_idx < func_block_ends[func_idx]) {
      /* If the number of get/cast/clear in this iteration is less than
      the recorded, reset the index to the recorded start position of
      the next function.
      */
      order_idx = func_block_ends[func_idx];
    }

    // Post process of the last function and swap out all arrays
    func_idx++;
    run_on_schedule();

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
                              const bool write_only, 
                              const bool first_creation) {
    synced_array_callback_tracer(saptr, sa_tag, dtype, ctx, write_only, 
                                 first_creation);};

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
  reconfirm_first_creation();

  bool do_reschedule = false;
  auto last_function = func_block_ends.size();

  // Using for prefetch cancel
  vector<unsigned int> prefetch_stopper(order.size(), 1);

  do {
    /* This counts the number of same arrays in the queue.
       If count of an array > 0, no need to fetch the same array again.
       If count of an array > 1, no need to swap out the array because it is
       planed to be used in the queue.
    */
    SyncedArrayStates synced_array_states;
    

    vector<bool> unprefetched(order.size(), false);

    int head = 0;
    int tail = 0;
    size_t used_bytes_swap_in = 0; // not including unprefetched arrays
    size_t prefetch_bytes = 0;     // including unprefetched arrays
    size_t used_bytes_swap_out = 0;
    unordered_map<unsigned int, bool> host_uses_this_synced_array;

    // They are used to remove uneccesary swap-out
    unordered_map<unsigned int, unordered_map<dtypes, bool>> swapped_out;
    unordered_map<unsigned int, RecType*> swapped_out_r;
    vector<RecType*> canceled_swap_out;

    // Preclear schedule will be used in swap-out schedule.
    schedule_preclear();

    // Virtually iterate all layer functions and solver update
    int fid = 0;

    // Calculate used GPU memory before forward starts,
    // and swap out if necessary.
    calc_mem_usage_before_forward(head, prefetch_bytes, used_bytes_swap_in,
                                  synced_array_states);
    schedule_swap_out(fid, prefetch_bytes, used_bytes_swap_in, used_bytes_swap_out,
                      synced_array_states, swapped_out, swapped_out_r);

    // Forward, backward, update
    for (fid = 1; fid < last_function; fid++) {
      schedule_swap_in(false, head, tail, fid, prefetch_bytes, used_bytes_swap_in,
                       used_bytes_swap_out, synced_array_states,
                       host_uses_this_synced_array, swapped_out, swapped_out_r,
                       canceled_swap_out, unprefetched, prefetch_stopper);

      do_reschedule =
        reserve_unprefetched_memory(head, tail, fid, prefetch_bytes, used_bytes_swap_in,
                                    used_bytes_swap_out, synced_array_states, 
                                    swapped_out, swapped_out_r, canceled_swap_out, 
                                    unprefetched, prefetch_stopper);

      if (do_reschedule) {
        break;
      }

      if (head < func_block_ends[fid]) {
        NBLA_ERROR(error_code::memory,
          "Some arrays were not prefetched probably due to out of GPU memory.");
      }

      schedule_swap_out(fid, prefetch_bytes, used_bytes_swap_in, used_bytes_swap_out,
                        synced_array_states, swapped_out, swapped_out_r);
    }

    if (do_reschedule) {
      continue;
    }

    fid = last_function - 1;
    schedule_wait_for_all_swap_out(fid, tail, used_bytes_swap_out, swapped_out,
                                   swapped_out_r, canceled_swap_out,
                                   synced_array_states);

    cancel_swap_out(canceled_swap_out);

    check_which_is_host_func();

    // The end of schedule
    do_reschedule = false;

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
backtrack_with_prefetch_cancel(int& head, const int fid,
                               const size_t unprefetched_bytes,
                               SyncedArrayStates& synced_array_states,
                               vector<unsigned int>& prefetch_stopper,
                               size_t available_bytes) {
  auto back_head = head;
  auto back_synced_array_states = synced_array_states;

  while (back_head >= func_block_ends[fid]) {
    back_head--; // decrement first because head indicates the next prefetch array.
    RecType *r = &order[back_head];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }

    if (r->ctx.array_class == device_ctx.array_class) {
      if (accumulate_counts(back_synced_array_states[r->said]) == 1) {
        // Release memory
        for (auto elem : back_synced_array_states[r->said]) {
          if (elem.second.state == ArrayStateTag::IN) {
            available_bytes += r->size * sizeof_dtype(elem.first);
          }
        }
        prefetch_stopper[back_head] = fid + 1;
      }

      if (available_bytes >= unprefetched_bytes) {
        // Ccanceled enough.
        break;
      }

      back_synced_array_states[r->said][r->dtype].count--;
    }
  }

  if (available_bytes < unprefetched_bytes) {
    NBLA_ERROR(error_code::memory, "A function is out of memory.");
  }
}


bool SwapInOutScheduler::
reserve_unprefetched_memory(int& head, int& tail, const int fid,
                            size_t& prefetch_bytes,
                            size_t& used_bytes_swap_in,
                            size_t& used_bytes_swap_out,
                            SyncedArrayStates& synced_array_states,
                            unordered_map<unsigned int, 
                                      unordered_map<dtypes, bool>>& swapped_out,
                            unordered_map<unsigned int, 
                                          RecType*>& swapped_out_r,
                            vector<RecType*>& canceled_swap_out,
                            vector<bool>& unprefetched,
                            vector<unsigned int>& prefetch_stopper) {
  size_t unprefetched_bytes = 0;

  for (auto i = func_block_ends[fid - 1]; i < func_block_ends[fid]; i++) {
    RecType *r = &order[i];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }

    if (r->ctx.array_class == device_ctx.array_class && unprefetched[i]) {
      unprefetched_bytes += r->size * sizeof_dtype(r->dtype);
    }
  }

  while (max_bytes - used_bytes_swap_in - used_bytes_swap_out
                                        < unprefetched_bytes) {
    if (tail == func_block_ends[fid - 1]) {
      // Out of memory, do backtrack with prefetch cancel and reschedule.
      auto available_bytes = max_bytes - used_bytes_swap_in - used_bytes_swap_out;
      backtrack_with_prefetch_cancel(head, fid, unprefetched_bytes,
                                     synced_array_states, prefetch_stopper,
                                     available_bytes);
      return true;
    }

    // Wait for swap out and release memory
    schedule_wait_for_swap_out_impl(fid, tail, used_bytes_swap_out,
                                    swapped_out, swapped_out_r,
                                    canceled_swap_out, synced_array_states);
  }

  // Memory for unprefetched arrays became available.
  used_bytes_swap_in += unprefetched_bytes;

  for (auto i = func_block_ends[fid - 1]; i < func_block_ends[fid]; i++) {
    RecType *r = &order[i];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }
    
    // UNPREFETCHED to IN
    if (r->ctx.array_class == device_ctx.array_class && unprefetched[i]) {
      if (synced_array_states[r->said][r->dtype].state != ArrayStateTag::UNPREFETCHED) {
        NBLA_ERROR(error_code::type, "UNPREFETCHED");
      }
      synced_array_states[r->said][r->dtype].state = ArrayStateTag::IN;
    }
  }

  return false;
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
calc_mem_usage_before_forward(int& head, size_t& prefetch_bytes,
                              size_t& used_bytes_swap_in,
                              SyncedArrayStates& synced_array_states) {
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
      if (synced_array_states[r->said][r->dtype].count == 0) {
        used_bytes_swap_in += array_bytes;
        prefetch_bytes += array_bytes;

        // CLEARED to IN
        if (synced_array_states[r->said][r->dtype].state != ArrayStateTag::CLEARED) {
          NBLA_ERROR(error_code::type, "CLEARED");
        }
        synced_array_states[r->said][r->dtype].state = ArrayStateTag::IN;
      }

      // Increment the number of the same SyncedArray in the queue.
      synced_array_states[r->said][r->dtype].count++;
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


void SwapInOutScheduler::
schedule_swap_in(const bool pre, int& head, int& tail, const int fid, 
                 size_t& prefetch_bytes,
                 size_t& used_bytes_swap_in, size_t& used_bytes_swap_out,
                 SyncedArrayStates& synced_array_states,
                 unordered_map<unsigned int, bool>& host_uses_this_synced_array,
                 unordered_map<unsigned int, unordered_map<dtypes, bool>>& swapped_out,
                 unordered_map<unsigned int, RecType*>& swapped_out_r,
                 vector<RecType*>& canceled_swap_out, 
                 vector<bool>& unprefetched, 
                 const vector<unsigned int> prefetch_stopper) {
  while (head < (pre ? func_block_ends[fid] : order.size())) {
    RecType *r = &order[head];

    if (r->tag == RecTag::CLEAR) {
      head++;
      continue;
    }
    
    if (r->ctx.array_class == device_ctx.array_class) {
      if (prefetch_stopper[head] > fid) {
        // Prefetch must be stopped to avoid out-of-memory in the future.
        break;
      }

      // Fetch
      if (synced_array_states[r->said][r->dtype].count == 0) {
        // The array is firstly appeared in the queue.
        auto next_array_bytes = r->size * sizeof_dtype(r->dtype);

        if (max_prefetch_bytes < prefetch_bytes + next_array_bytes) {
          // Out of prefetch memory
          break;
        }
       
        if (!host_uses_this_synced_array[r->said]) {
          // If the array is scheduled to be swapped out,
          // by canceling it, this prefetch can be omitted.
          if (swapped_out[r->said][r->dtype]) {
            auto sor = swapped_out_r[r->said];

            // Cancel the swap out
            canceled_swap_out.push_back(sor);

            // Remove the array sizes from swap-out memory
            sor->swapped_out = false;
            used_bytes_swap_out -= sor->swapped_out_bytes;
            next_array_bytes = sor->swapped_out_bytes;
            sor->swapped_out_bytes = 0;

            // Reset flags
            swapped_out[r->said].clear();
            swapped_out_r[r->said] = nullptr;

            // Increase memory usage
            used_bytes_swap_in += next_array_bytes;

            for (auto& elem : synced_array_states[r->said]) {
              // OUT to IN
              if (elem.second.state == ArrayStateTag::OUT) {
                elem.second.state = ArrayStateTag::IN;
              }
            }
          }
          else if (no_data_transfer(r)) {
            // Prefetch is unnecessary because data transfer will not happen.
            unprefetched[head] = true;

            // CLEARED -> UNPREFETCHED
            if (synced_array_states[r->said][r->dtype].state != ArrayStateTag::CLEARED) {
              NBLA_ERROR(error_code::type, "CLEARED");
            }
            synced_array_states[r->said][r->dtype].state = ArrayStateTag::UNPREFETCHED;
          }
          else {
            // Swap out for prefetch
            bool no_memory = false;

            while (used_bytes_swap_in + used_bytes_swap_out
                                      + next_array_bytes > max_bytes) {
              if (tail == func_block_ends[fid - 1]) {
                no_memory = true;
                break;
              }

              // Out of memory
              // Wait for swap out and release memory
              schedule_wait_for_swap_out_impl(fid, tail, used_bytes_swap_out,
                                              swapped_out, swapped_out_r,
                                              canceled_swap_out, synced_array_states);
            }

            if (no_memory) {
              break;
            }

            schedules_swap[fid].push_back(ScheduleType(ScheduleTag::SWAP_IN, r));

            // Increase memory usage
            used_bytes_swap_in += next_array_bytes;

            // CLEARED or OUT_WAITED to IN
            if (synced_array_states[r->said][r->dtype].state != ArrayStateTag::CLEARED && 
                synced_array_states[r->said][r->dtype].state != ArrayStateTag::OUT_WAITED) {
              NBLA_ERROR(error_code::type, "CLEARED");
            }
            synced_array_states[r->said][r->dtype].state = ArrayStateTag::IN;
          }
        }
        else {
          // Increase memory usage
          used_bytes_swap_in += next_array_bytes;

          // Firstly HOSTE_USED to IN
          synced_array_states[r->said][r->dtype].state = ArrayStateTag::IN;
        }

        // Count prefetch bytes regardless of prefetched or unprefetched.
        prefetch_bytes += next_array_bytes;
      }

      // Increment the number of the same SyncedArray in the queue.
      synced_array_states[r->said][r->dtype].count++;
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

        /*
        // CLEAERED or OUT_WAITED to HOST_USED
        if (synced_array_states[r->said][r->dtype].state != ArrayStateTag::CLEARED &&
            synced_array_states[r->said][r->dtype].state != ArrayStateTag::HOST_USED) {
          NBLA_ERROR(error_code::type, "CLEARED or HOST_USED");
        }

        synced_array_states[r->said][r->dtype].state = ArrayStateTag::HOST_USED;
        */
      }

      // The arrray comes on to host by swap out.
      if (swapped_out[r->said][r->dtype]) {
        // reset flag
        swapped_out[r->said].clear();
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
}


void SwapInOutScheduler::
schedule_swap_out(const int fid, size_t& prefetch_bytes,
                  size_t& used_bytes_swap_in, size_t& used_bytes_swap_out,
                  SyncedArrayStates& synced_array_states, 
                  unordered_map<unsigned int, unordered_map<dtypes, bool>>& swapped_out,
                  unordered_map<unsigned int, RecType*>& swapped_out_r) {
  for (size_t i = (fid == 0 ? 0 : func_block_ends[fid - 1]);
              i < func_block_ends[fid];
              i++) {
    RecType *r = &order[i];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }

    if (r->ctx.array_class == device_ctx.array_class) { 
      if (accumulate_counts(synced_array_states[r->said]) == 1) {
        // An array is swapped out when the same array is no longer
        // in the queue.
        if (std::find(preclear_schedule[fid].begin(), 
                      preclear_schedule[fid].end(), r)
                      == preclear_schedule[fid].end()) {
          // Not precleared
          schedules_swap[fid + 1].push_back(ScheduleType(ScheduleTag::SWAP_OUT, r));

          r->swapped_out = true;
          swapped_out_r[r->said] = &order[i];
        
          // Transfer memory usage of all types
          r->swapped_out_bytes = 0;

          for (auto& elem : synced_array_states[r->said]) {           
            // IN to OUT
            if (elem.second.state == ArrayStateTag::IN) {
              swapped_out[r->said][elem.first] = true;

              auto array_bytes = r->size * sizeof_dtype(elem.first);
              used_bytes_swap_out += array_bytes;
              r->swapped_out_bytes += array_bytes;

              used_bytes_swap_in -= array_bytes;
              prefetch_bytes -= array_bytes;
              elem.second.state = ArrayStateTag::OUT;
            }
          }
        }
        else {
          // Any states to CLEARED
          for (auto& elem : synced_array_states[r->said]) {
            if (elem.second.state == ArrayStateTag::IN) {
              auto array_bytes = r->size * sizeof_dtype(elem.first);
              used_bytes_swap_in -= array_bytes;
              prefetch_bytes -= array_bytes;
            }

            elem.second.state = ArrayStateTag::CLEARED;
          }
        }

        // Reset usage
        //synced_array_states[r->said].clear();
        synced_array_states[r->said][r->dtype].count--;
      }
      else {
        // Decrease the counts of a used array in the queue.
        synced_array_states[r->said][r->dtype].count--;
      }
    }
    else if (r->ctx.array_class != host_ctx.array_class) {
      // Get/cast of an array on an uncertain device
      NBLA_ERROR(error_code::type, 
                 "Unsupported array type: " + r->ctx.array_class);
    }
  }
}


void SwapInOutScheduler::schedule_wait_for_all_swap_out(
  const int fid, int& tail, size_t& used_bytes_swap_out,
  unordered_map<unsigned int, unordered_map<dtypes, bool>>& swapped_out,
  unordered_map<unsigned int, RecType*>& swapped_out_r,
  vector<RecType*>& canceled_swap_out,
  SyncedArrayStates& synced_array_states)
{
  // When out of memory, wait for finishing swap out.
  while (tail < order.size()) {
    schedule_wait_for_swap_out_impl(fid, tail, used_bytes_swap_out,
                                    swapped_out, swapped_out_r, 
                                    canceled_swap_out, synced_array_states);
  }
}


void SwapInOutScheduler::schedule_wait_for_swap_out_impl(
  const int fid, int& tail, size_t& used_bytes_swap_out,
  unordered_map<unsigned int, unordered_map<dtypes, bool>>& swapped_out,
  unordered_map<unsigned int, RecType*>& swapped_out_r,
  vector<RecType*>& canceled_swap_out,
  SyncedArrayStates& synced_array_states)
{
  RecType *r = &order[tail++];

  if (r->swapped_out && std::find(canceled_swap_out.begin(), 
                                  canceled_swap_out.end(),
                                  r) == canceled_swap_out.end()) {
    // Not canceled swap out
    // Wait for finishing swap out and release the source array of memory copy.
    schedules_swap[fid + 1].push_back(ScheduleType(ScheduleTag::WAIT, r));

    // Decrease memory usage
    r->swapped_out = false;
    used_bytes_swap_out -= r->swapped_out_bytes;
    r->swapped_out_bytes = 0;

    swapped_out[r->said].clear();
    swapped_out_r[r->said] = nullptr;

    for (auto& elem : synced_array_states[r->said]) {
      if (elem.second.state == ArrayStateTag::OUT) {
        elem.second.state = ArrayStateTag::OUT_WAITED;
      }
    }
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


void SwapInOutScheduler::cancel_swap_out(vector<RecType*>& canceled_swap_out) {
  for (int fid = 0; fid < func_block_ends.size(); fid++) {
    for (auto it = schedules_swap[fid].begin();
              it != schedules_swap[fid].end();) {
      if (it->tag == ScheduleTag::SWAP_OUT &&
          std::find(canceled_swap_out.begin(),
                    canceled_swap_out.end(),
                    it->r) != canceled_swap_out.end()) {
        it = schedules_swap[fid].erase(it);
      }
      else {
        ++it;
      }
    }
  }
}


//----------------------------------------------------------------
//  Execute swap in/out
//----------------------------------------------------------------
// Common implementation of pre callback
void SwapInOutScheduler::pre_callback() {
  unset_synced_array_callback(); // Avoid unnecessary record and trace

  if (first_iter) {
    // Record the end of a function.
    func_block_ends.push_back(order_idx);

    // Swap out and preclear the arrays used in the previous function.
    swap_out_first_iter();
    wait_for_swap_out_first_iter();

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

    func_idx++;
    run_on_schedule();

    if (is_host_func[func_idx]) {
      BackendUtils::default_stream_synchronize(device_ctx);
    }
  }

  set_synced_array_callback(); // Restart record or trace
}


// Common implementation of post callback
void SwapInOutScheduler::post_callback() {
  if (!first_iter) {
    unset_synced_array_callback(); // Avoid unnecessary record and trace
    for (auto r : preclear_schedule[func_idx]) {
      if (auto p = r->sawptr.lock()) {
        p->clear();
        precleared[p] = true;
      }
    }
    set_synced_array_callback(); // Restart record or trace
  }
}


void SwapInOutScheduler::run_on_schedule() {
  for (const auto& s : schedules_swap[func_idx]) {
    if (s.tag == ScheduleTag::SWAP_IN) {
      if (auto p = s.r->sawptr.lock()) {
        p->get(s.r->dtype, device_ctx, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
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
  }  
}


// In the first iteration, arrays used in a function are always swapped out.
void SwapInOutScheduler::swap_out_first_iter() {
  // Counts SyncedArrays which were used in the previous function.
  SyncedArrayStates synced_array_states;
  const int start_idx = func_idx == 0 ? 0 : func_block_ends[func_idx - 1];

  for (int i = start_idx; i < func_block_ends[func_idx]; i++) {
    RecType *r = &order[i];
    if (r->tag == RecTag::CLEAR) continue;

    if (r->ctx.array_class == device_ctx.array_class) {
      synced_array_states[r->said][r->dtype].count++;
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
      if (accumulate_counts(synced_array_states[r->said]) == 1) {
        auto p = r->sawptr.lock();

        if (p && is_not_cleared_yet(p)) {
          // The array is not cleared yet. Swap out the array
          p->cast(p->dtype(), host_ctx, false, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
          r->swapped_out = true;

          // Counts memory usage of all types
          r->swapped_out_bytes = 0;

          for (auto it : synced_array_states[r->said]) {
            auto array_bytes = r->size * sizeof_dtype(it.first);
            used_bytes_swap_out_first_iter += array_bytes;
            r->swapped_out_bytes += array_bytes;
          }
        }

        synced_array_states[r->said].clear();
      }
      else {
        synced_array_states[r->said][r->dtype].count--;
      }
    }
  }
}

// It is necessary to prefetch arrays of a single function at the first iter
// And wait for swapped out.
void SwapInOutScheduler::wait_for_swap_out_first_iter() {
  while (used_bytes_swap_out_first_iter > max_bytes / 2) { // temporary!!!!!!
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
                               const bool write_only,
                               const bool first_creation) {
  auto tag = convert_tag(sa_tag, write_only);

  // Define SyncedArray ID
  if (said_map.find(saptr) == said_map.end()) {
    said_map[saptr] = static_cast<unsigned int>(said_map.size());
  }
  auto said = said_map.at(saptr);

  // Record the order
  order.push_back(RecType{tag, said, saptr, saptr->size(), dtype, ctx,
                          write_only, first_creation});

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
                             const bool write_only,
                             const bool first_creation) {
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
