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

using std::cout;
using std::endl;

using std::accumulate;

// Constructor
SwapInOutScheduler::SwapInOutScheduler(const Context &h_ctx,
                                       const Context &d_ctx, const size_t max,
                                       const size_t prefetch_max,
                                       const bool save_host_mem,
                                       const bool save_host_mem_no_abort)
    : host_ctx(h_ctx), device_ctx(d_ctx), max_bytes(max),
      max_prefetch_bytes(prefetch_max == 0 ? max * 1.5 : prefetch_max),
      cast_prefetch(save_host_mem),
      cast_prefetch_no_abort(save_host_mem_no_abort),
      free_host_caches(save_host_mem),
      sa_callback([&](SyncedArrayPtr saptr, const SyncedArrayCallbackTag sa_tag,
                      const dtypes dtype, const Context &ctx,
                      const bool write_only, const bool first_creation,
                      const bool off_recording) {
        // Set SyncedArrayCallback function for first iteration
        sa_callback_recorder(saptr, sa_tag, dtype, ctx, write_only,
                             first_creation, off_recording);
      }) {
  // Create non blocking streams for data transfer
  BackendUtils::create_lms_streams(d_ctx);
}

// Destructor
SwapInOutScheduler::~SwapInOutScheduler() {}

// User interface to start a scheduling code block
void SwapInOutScheduler::start_scheduling() {
  if (second_iter) {
    schedule();       // Schedule swap in/out
    said_map.clear(); // Clear variables used in the first iteration

    // Free host caches allocated too much before scheduled execution.
    if (free_host_caches) {
      BackendUtils::free_unused_host_caches(host_ctx);
      BackendUtils::free_unused_host_caches(device_ctx);
    }
  }

  // Init
  order_idx = 0;
  func_idx = 0;
  wrong_ordered.clear();
  precleared.clear();
  cast_prefetched.clear();
  set_sa_callback(); // Set SyncedArrayCallback
}

// User interface to finish a scheduling code block
void SwapInOutScheduler::end_scheduling() {
  unset_sa_callback(); // Unset a SyncedArrayCallback

  // Post process of the last function.
  if (first_iter) {
    func_block_ends.push_back(order_idx); // Record the end of a function.
    swap_out_first_iter(); // Swap out the arrays of the last function
  } else {
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
  sa_callback = [&](SyncedArrayPtr saptr, const SyncedArrayCallbackTag sa_tag,
                    const dtypes dtype, const Context &ctx,
                    const bool write_only, const bool first_creation,
                    const bool off_recording) {
    sa_callback_tracer(saptr, sa_tag, dtype, ctx, write_only, first_creation,
                       off_recording);
  };

  if (first_iter) {
    first_iter = false;
    second_iter = true;
  } else if (second_iter) {
    second_iter = false;
  }
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

void SwapInOutScheduler::pre_update_callback() { pre_callback(); }

void SwapInOutScheduler::post_update_callback() { post_callback(); }

//----------------------------------------------------------------
//  Scheduler
//----------------------------------------------------------------
void SwapInOutScheduler::determine_first_head_types(
    unordered_map<unsigned int, pair<bool, dtypes>> &head_type) {
  for (const auto &r : order) {
    auto p = r.sawptr.lock();
    if (p && is_not_cleared_yet(p)) {
      head_type[r.said] = std::make_pair(true, p->dtype());
    } else {
      head_type[r.said] = std::make_pair(false, dtypes::BYTE);
    }
  }
}

void SwapInOutScheduler::collect_info_about_dtype_conversion(
    unordered_map<unsigned int, bool> &type_converted,
    const unordered_map<unsigned int, pair<bool, dtypes>> &head_type) {
  unordered_map<unsigned int, set<dtypes>> counter;

  // Count all dtypes in the recorded order.
  for (const auto &r : order) {
    counter[r.said].insert(r.dtype);
  }

  // Count all dtypes before scheduling.
  // We need to check both order and head_typd,
  // since order and head_type could have different dtypes.
  // If user touch some arraies outside of nnabla's graph engine,
  // syncedArray would have other dtypes which are not recorded by
  // sa_callback_tracer().
  for (const auto &h : head_type) {
    if (h.second.first) {
      counter[h.first].insert(h.second.second);
    }
  }

  for (const auto &c : counter) {
    if (c.second.size() > 1) {
      type_converted[c.first] = true;
    }
  }
}

struct bytes_state {
  size_t swap_in, swap_out, prefetch;
  bytes_state(size_t i, size_t o, size_t p)
      : swap_in(i), swap_out(o), prefetch(p) {}
};

void SwapInOutScheduler::schedule() {
  reconfirm_first_creation();

  // Determine the first dtype of head array.
  unordered_map<unsigned int, pair<bool, dtypes>> first_head_type;
  unordered_map<unsigned int, bool> type_converted;
  determine_first_head_types(first_head_type);
  collect_info_about_dtype_conversion(type_converted, first_head_type);
  unordered_map<unsigned int, pair<bool, dtypes>> head_type = first_head_type;

  bool do_reschedule = false;
  auto last_function = func_block_ends.size();

  // Using for prefetch cancel
  vector<unsigned int> prefetch_stopper(order.size(), 1);

  // for debugging
  vector<bytes_state> bytes_debugger;

  do {
    // Reset schedules
    beginning_schedules.clear();
    end_schedules.clear();
    beginning_schedules.resize(last_function +
                               1); // +1 is for the last "wait for all"
    end_schedules.resize(last_function);
    head_type = first_head_type;
    ScheduleParams params;

    bytes_debugger.clear();

    // Preclear schedule will be used in swap-out schedule.
    unordered_map<unsigned int, vector<pair<RecType *, bool>>> clear_info;
    schedule_preclear(clear_info);

    // Calculate used GPU memory before forward starts, and swap out if
    // necessary.
    calc_mem_usage_before_forward(params, head_type);

    // Forward, backward, update
    for (params.fid = 1; params.fid < last_function; params.fid++) {
      schedule_swap_in(params, prefetch_stopper, type_converted, head_type);
      do_reschedule = reserve_unprefetched_memory(params, prefetch_stopper);

      if (do_reschedule) {
        break;
      }

      if (params.head < func_block_ends[params.fid]) {
        NBLA_ERROR(error_code::memory, "Some arrays were not prefetched "
                                       "probably due to out of GPU memory.");
      }

      // Check the change of head array type just in Function.
      for (size_t i = func_block_ends[params.fid - 1];
           i < func_block_ends[params.fid]; i++) {
        RecType *r = &order[i];
        auto &htype = head_type[r->said];

        if (r->tag == RecTag::CLEAR) {
          // Remove head dtype
          htype.first = false;
        } else if (htype.first) {
          if (r->tag == RecTag::CAST) {
            // Update head array type
            htype.second = r->dtype;
          }
        } else {
          // The array was fetched firstly.
          htype.first = true;
          htype.second = r->dtype;
        }
      }

      if (params.fid == 1) {
        // Swap out arrays before first prefetch if needed
        params.fid = 0;
        schedule_swap_out(params, clear_info, head_type);
        params.fid = 1;
      }
      schedule_swap_out(params, clear_info, head_type);

      bytes_debugger.emplace_back(params.swap_in_bytes, params.swap_out_bytes,
                                  params.prefetch_bytes);
    }

    if (do_reschedule) {
      continue;
    }

    schedule_wait_for_all_swap_out(params);
    determine_which_is_host_func();

    // The end of schedule
    do_reschedule = false;

#if 0
    for (int i = 0; i < bytes_debugger.size(); i++) {
      auto e = bytes_debugger[i];
      printf("[fid :%d] SI: %s SO: %s PF: %s\n",
              i,
              byte_to_human_readable(e.swap_in).c_str(),
              byte_to_human_readable(e.swap_out).c_str(),
              byte_to_human_readable(e.prefetch).c_str());
      fflush(stdout);
    }
#endif

// Debug
#if SWAPINOUTSCHEDULER_DEBUG
    for (const auto &s : params.sa_states) {
      for (const auto &elem : s.second) {
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
  } while (do_reschedule);
}

/* Recorded array creations could be fake in the first iteration.
   This method confirms true those first creation.
*/
void SwapInOutScheduler::reconfirm_first_creation() {
  unordered_map<unsigned int, bool> cleared;

  for (auto &r : order) {
    if (r.tag == RecTag::CLEAR) {
      cleared[r.said] = true;
    } else {
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

bool SwapInOutScheduler::no_data_transfer(const RecType *r) {
  return (r->write_only_cast || r->first_creation);
}

int SwapInOutScheduler::accumulate_counts(
    const unordered_map<dtypes, ArrayState> &count_map) {
  return accumulate(
      count_map.begin(), count_map.end(), 0,
      [](int value, const unordered_map<dtypes, ArrayState>::value_type &p) {
        return value + p.second.count;
      });
}

void SwapInOutScheduler::backtrack_with_prefetch_cancel(
    ScheduleParams &params, vector<unsigned int> &prefetch_stopper,
    const size_t unprefetched_bytes, size_t available_bytes) {
  auto back_head = params.head;
  auto back_sa_states = params.sa_states;

  while (back_head >= func_block_ends[params.fid]) {
    back_head--; // decrement first because head indicates the next prefetch
                 // array.
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
      // Canceled enough.
      break;
    }

    back_sa_states[r->said][r->dtype].count--;
  }

  if (available_bytes < unprefetched_bytes) {
    NBLA_ERROR(error_code::memory, "A function is out of memory.");
  }
}

bool SwapInOutScheduler::reserve_unprefetched_memory(
    ScheduleParams &params, vector<unsigned int> &prefetch_stopper) {
  // Count unprefetched bytes only once
  unordered_map<unsigned int, unordered_map<dtypes, size_t>>
      uniq_unprefetched_bytes;

  for (auto i = func_block_ends[params.fid - 1];
       i < func_block_ends[params.fid]; i++) {
    RecType *r = &order[i];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }

    if (params.sa_states[r->said][r->dtype].state ==
        ArrayStateTag::UNPREFETCHED) {
      uniq_unprefetched_bytes[r->said][r->dtype] =
          r->size * sizeof_dtype(r->dtype);
    }
  }

  size_t unprefetched_bytes = 0;
  for (const auto &sa_bytes : uniq_unprefetched_bytes) {
    for (const auto bytes : sa_bytes.second) {
      unprefetched_bytes += bytes.second;
    }
  }

  while (max_bytes - params.swap_in_bytes - params.swap_out_bytes <
         unprefetched_bytes) {
    if (params.tail == func_block_ends[params.fid - 1]) {
      // Out of memory, do backtrack with prefetch cancel and reschedule.
      auto available_bytes =
          max_bytes - params.swap_in_bytes - params.swap_out_bytes;
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
    if (params.sa_states[r->said][r->dtype].state ==
        ArrayStateTag::UNPREFETCHED) {
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

void SwapInOutScheduler::calc_mem_usage_before_forward(
    ScheduleParams &params,
    unordered_map<unsigned int, pair<bool, dtypes>> &head_type) {

  for (; params.head < func_block_ends[0]; params.head++) {
    RecType *r = &order[params.head];

    if (r->tag == RecTag::CLEAR)
      continue;

    // Check the change of head array type
    // The changes were already done before first prefetch.
    auto &htype = head_type[r->said];
    if (htype.first) {
      if (r->tag == RecTag::CAST) {
        // Update head array type
        htype.second = r->dtype;
      }
    } else {
      // The array was fetched firstly.
      htype.first = true;
      htype.second = r->dtype;
    }

    // First fetch
    if (params.sa_states[r->said][r->dtype].count == 0) {
      // All fetches were already done. Just calculate memory size.
      auto array_bytes = r->size * sizeof_dtype(r->dtype);

      params.swap_in_bytes += array_bytes;
      params.prefetch_bytes += array_bytes;

      // CLEARED to IN
      // note: This condition cannot be happended. Could delete this safely.
      if (params.sa_states[r->said][r->dtype].state != ArrayStateTag::CLEARED) {
        NBLA_ERROR(error_code::type,
                   "Array state must be CLEARED before first fetch.");
      }
      params.sa_states[r->said][r->dtype].state = ArrayStateTag::IN;
    }

    // Increment the number of the same SyncedArray in the queue.
    params.sa_states[r->said][r->dtype].count++;
  }
}

void SwapInOutScheduler::cancel_swap_out(const RecType *r,
                                         ScheduleParams &params) {
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
  }();

  size_t bytes = 0;
  for (auto &elem : params.sa_states[r->said]) {
    if (elem.second.state == ArrayStateTag::OUT ||
        elem.second.state == ArrayStateTag::OUT_CLEARED) {
      bytes += r->size * sizeof_dtype(elem.first);

      if (elem.second.state == ArrayStateTag::OUT) {
        params.swap_out_bytes -= r->size * sizeof_dtype(elem.first);
      }

      elem.second.state = ArrayStateTag::IN; // OUT and OUT_CLEARED to IN
    }
  }

  params.swap_in_bytes += bytes;
  params.prefetch_bytes += bytes;
}

bool SwapInOutScheduler::free_memory_to_prefetch(ScheduleParams &params,
                                                 const size_t array_bytes) {
  bool no_memory = false;

  while (params.swap_in_bytes + params.swap_out_bytes + array_bytes >
         max_bytes) {
    if (params.tail == func_block_ends[params.fid - 1]) {
      no_memory = true;
      break;
    }

    // Out of memory. Wait for swap out and release memory
    schedule_wait_for_swap_out_impl(params);
  }

  return no_memory;
}

void SwapInOutScheduler::schedule_swap_in(
    ScheduleParams &params, const vector<unsigned int> prefetch_stopper,
    unordered_map<unsigned int, bool> &type_converted,
    unordered_map<unsigned int, pair<bool, dtypes>> &head_type) {
  while (params.head < order.size()) {
    RecType *r = &order[params.head];

    if (r->tag == RecTag::CLEAR) {
      params.head++;
      continue;
    }

    // Prefetch must be stopped to avoid out-of-memory in the future.
    if (prefetch_stopper[params.head] > params.fid)
      break;

    if (params.sa_states[r->said][r->dtype].count == 0) {
      // The array is firstly appeared in the queue.
      const auto array_bytes = r->size * sizeof_dtype(r->dtype);

      // Out of prefetch memory
      if (max_prefetch_bytes < params.prefetch_bytes + array_bytes ||
          max_bytes < params.swap_in_bytes + array_bytes) {
        break;
      }

      if (context_checker(r->ctx, device_ctx)) {
        if (params.sa_states[r->said][r->dtype].state == ArrayStateTag::OUT ||
            params.sa_states[r->said][r->dtype].state ==
                ArrayStateTag::OUT_CLEARED) {
          // Swap out cancel
          cancel_swap_out(r, params);
        } else if (no_data_transfer(r)) { // Unprefetch due to no data transfer
          params.prefetch_bytes += array_bytes;

          // CLEARED -> UNPREFETCHED
          params.sa_states[r->said][r->dtype].state =
              ArrayStateTag::UNPREFETCHED;
        } else {                        // Prefetch
          size_t extra_array_bytes = 0; // Temporary buffer of type conversion

          // Add the temporary buffer size if array type will be converted
          // when prefetch.
          auto &htype = head_type[r->said];
          if (htype.first && htype.second != r->dtype) {
            extra_array_bytes = r->size * sizeof_dtype(htype.second);
          }

          if (max_bytes <
              params.swap_in_bytes + array_bytes + extra_array_bytes) {
            break;
          }

          // Free memory to prefetch the next array
          free_memory_to_prefetch(params, array_bytes + extra_array_bytes);

          if (cast_prefetch &&
              (!type_converted[r->said] ||
               (r->tag == RecTag::CAST &&
                accumulate_counts(params.sa_states[r->said]) == 0))) {
            // Prefetch using cast when the type of this array is not converted
            // through the recorded order, or when this array is fetched firstly
            // by cast().
            beginning_schedules[params.fid].push_back(
                ScheduleType(ScheduleTag::SWAP_IN_CAST, r));

            // Update head array type
            htype.first = true;
            htype.second = r->dtype;
          } else {
            beginning_schedules[params.fid].push_back(
                ScheduleType(ScheduleTag::SWAP_IN_GET, r));

            // Update head array type
            if (!htype.first) {
              htype.second = r->dtype;
            }
          }

          params.swap_in_bytes += array_bytes; // Increase memory usage
          params.prefetch_bytes += array_bytes;

          // CLEARED or OUT_WAITED to IN
          params.sa_states[r->said][r->dtype].state = ArrayStateTag::IN;
        }
      } else { // Host array
        // Do not prefetch, Do swap out
        if (params.sa_states[r->said][r->dtype].state == ArrayStateTag::OUT ||
            params.sa_states[r->said][r->dtype].state ==
                ArrayStateTag::OUT_CLEARED) {
          // Decrease bytes of swap out
          size_t bytes = 0;
          for (auto &elem : params.sa_states[r->said]) {
            if (elem.second.state == ArrayStateTag::OUT) {
              bytes += r->size * sizeof_dtype(elem.first);
              elem.second.state =
                  ArrayStateTag::IN; // OUT and OUT_CLEARED to IN
            } else if (elem.second.state == ArrayStateTag::OUT_CLEARED) {
              elem.second.state = ArrayStateTag::CLEARED;
            }
          }

          params.swap_out_bytes -= bytes;
          params.swap_in_bytes += bytes;
          params.prefetch_bytes += bytes;
        } else {
          // Free memory to prefetch the next array
          free_memory_to_prefetch(params, array_bytes);

          params.swap_in_bytes += array_bytes; // Increase memory usage
          params.prefetch_bytes += array_bytes;

          // CLEARED or OUT_WAITED to IN
          params.sa_states[r->said][r->dtype].state = ArrayStateTag::IN;
        }
      }
    }

    // Increment the number of the same SyncedArray in the queue.
    params.sa_states[r->said][r->dtype].count++;
    params.head++; // Move on the head of the queue
  }
}

void SwapInOutScheduler::schedule_swap_out(
    ScheduleParams &params,
    unordered_map<unsigned int, vector<pair<RecType *, bool>>> &clear_info,
    unordered_map<unsigned int, pair<bool, dtypes>> &head_type) {
  for (size_t i = (params.fid == 0 ? 0 : func_block_ends[params.fid - 1]);
       i < func_block_ends[params.fid]; i++) {
    RecType *r = &order[i];

    if (r->tag == RecTag::CLEAR) {
      continue;
    }

    if (accumulate_counts(params.sa_states[r->said]) == 1) {
      // An array is swapped out when the same array is no longer
      // in the queue.
      auto do_preclear = std::find_if(
          clear_info[params.fid].begin(), clear_info[params.fid].end(),
          [&](pair<RecType *, bool> x) { return x.first == r; });

      if (do_preclear != clear_info[params.fid].end()) { // Cleared
        for (auto &elem : params.sa_states[r->said]) { // Any states to CLEARED
          if (elem.second.state == ArrayStateTag::IN) {
            auto array_bytes = r->size * sizeof_dtype(elem.first);
            params.swap_in_bytes -= array_bytes;
            params.prefetch_bytes -= array_bytes;
          }

          elem.second.state = ArrayStateTag::CLEARED;
        }

        if (do_preclear->second) {
          end_schedules[params.fid].push_back(
              ScheduleType(ScheduleTag::PRECLEAR, r));

          // Update head array type
          head_type[r->said].first = false;
        }
      } else if (r->temporary_buffer) {
        for (auto &elem : params.sa_states[r->said]) { // Any states to CLEARED
          if (elem.second.state == ArrayStateTag::IN) {
            auto array_bytes = r->size * sizeof_dtype(elem.first);
            params.swap_in_bytes -= array_bytes;
            params.prefetch_bytes -= array_bytes;
          }

          elem.second.state = ArrayStateTag::CLEARED;
        }

        // Update head array type
        head_type[r->said].first = false;
      } else { // Not precleared, Swap out
        end_schedules[params.fid].push_back(
            ScheduleType(ScheduleTag::SWAP_OUT, r));

        // Transfer memory usage of all types
        for (auto &elem : params.sa_states[r->said]) {
          if (elem.second.state == ArrayStateTag::IN) {
            // Swap out is performed by cast.
            // Thus, all dtypes are cleared from device.
            auto array_bytes = r->size * sizeof_dtype(elem.first);
            params.swap_in_bytes -= array_bytes;
            params.prefetch_bytes -= array_bytes;
            elem.second.swapped_out_r = r;

            if (elem.first == head_type[r->said].second) {
              // Swap out only head
              params.swap_out_bytes += array_bytes;
              elem.second.state = ArrayStateTag::OUT; // IN to OUT
            } else {
              // Do not add array_bytes to swap_out_bytes
              elem.second.state =
                  ArrayStateTag::OUT_CLEARED; // IN to OUT_CLREARED
            }
          }
        }
      }
    }

    // Decrease the counts of a used array in the queue.
    params.sa_states[r->said][r->dtype].count--;
  }
}

void SwapInOutScheduler::schedule_wait_for_all_swap_out(
    ScheduleParams &params) {
  // When out of memory, wait for finishing swap out.
  while (params.tail < order.size()) {
    schedule_wait_for_swap_out_impl(params);
  }
}

void SwapInOutScheduler::schedule_wait_for_swap_out_impl(
    ScheduleParams &params) {
  RecType *r = &order[params.tail];

  if ((params.sa_states[r->said][r->dtype].state == ArrayStateTag::OUT ||
       params.sa_states[r->said][r->dtype].state ==
           ArrayStateTag::OUT_CLEARED) &&
      params.sa_states[r->said][r->dtype].swapped_out_r == r) {
    // Not canceled swap out
    // Wait for finishing swap out and release the source array of memory copy.
    beginning_schedules[params.fid].push_back(
        ScheduleType(ScheduleTag::WAIT, r));

    // Decrease memory usage
    size_t bytes = 0;
    for (auto &elem : params.sa_states[r->said]) {
      if (elem.second.state == ArrayStateTag::OUT) {
        bytes += r->size * sizeof_dtype(elem.first);
        elem.second.state = ArrayStateTag::OUT_WAITED; // OUT to OUT_WAITED
      } else if (elem.second.state == ArrayStateTag::OUT_CLEARED) {
        elem.second.state = ArrayStateTag::CLEARED; // OUT_CLEARED to CLEARED
      }
    }
    params.swap_out_bytes -= bytes;
    params.sa_states[r->said][r->dtype].swapped_out_r = nullptr;
  }

  params.tail++;
}

void SwapInOutScheduler::schedule_preclear(
    unordered_map<unsigned int, vector<pair<RecType *, bool>>> &clear_info) {
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
      } else {
        if (clear_last.find(r->said) == clear_last.end()) {
          clear_last[r->said] = false;
        }

        if (clear_trace[r->said].first) { // get/cast just before clear
          if (fid < clear_trace[r->said].second) {
            // Preclear
            clear_info[fid].emplace_back(r, true);
          } else if (clear_last.at(r->said)) {
            // Do not preclear because clear before preclear.
            clear_info[fid].emplace_back(r, false);
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
  } else {
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
      BackendUtils::device_synchronize(device_ctx);
    }
  }

  set_sa_callback(); // Restart record or trace
}

// Common implementation of post callback
void SwapInOutScheduler::post_callback() {}

void SwapInOutScheduler::run(const ScheduleType &s) {
  if (auto p = s.r->sawptr.lock()) {
#if 0
    cout << "type: " << to_str(s.tag);
    cout << " said: " << s.r->said;
    cout << " saptr: " << (uint64_t) s.r->sawptr.lock().get();
    cout << " bytes: " << byte_to_human_readable(s.r->size *
    sizeof_dtype(s.r->dtype));
    cout << " dtype: " << dtype_to_string(s.r->dtype) << endl;
    cout << " count: " << s.r->sawptr.use_count() << endl;
#endif

    if (s.tag == ScheduleTag::SWAP_IN_GET) {
      p->get(s.r->dtype, device_ctx, AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
    } else if (s.tag == ScheduleTag::SWAP_IN_CAST) {
      p->cast(s.r->dtype, device_ctx, false,
              AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
      cast_prefetched[p] = true;
    } else if (s.tag == ScheduleTag::SWAP_OUT) {
      // Swap out and preclear the arrays used in the previous function.
      if (is_not_cleared_yet(p)) {
        p->cast(p->dtype(), host_ctx, false,
                AsyncFlag::ASYNC | AsyncFlag::UNSAFE);
      }
    } else if (s.tag == ScheduleTag::WAIT) {
      if (p->head_array_class() == host_ctx.array_class &&
          is_not_cleared_yet(p)) {
        p->get(p->dtype(), host_ctx, AsyncFlag::UNSAFE);
      }
    } else if (s.tag == ScheduleTag::PRECLEAR) {
      p->clear();
      precleared[p] = true;
    }
  }
}

inline string SwapInOutScheduler::to_str(const ScheduleTag &st) {
  if (st == ScheduleTag::PRECLEAR)
    return "Preclear";
  if (st == ScheduleTag::SWAP_IN_CAST)
    return "SwapInCast";
  if (st == ScheduleTag::SWAP_IN_GET)
    return "SwapInGet";
  if (st == ScheduleTag::SWAP_OUT)
    return "SwapOut";
  if (st == ScheduleTag::WAIT)
    return "Wait";
}

void SwapInOutScheduler::run_on_beginning_schedule() {
  for (const auto &s : beginning_schedules[func_idx]) {
    run(s);
  }
}

void SwapInOutScheduler::run_on_end_schedule() {
  for (const auto &s : end_schedules[func_idx]) {
    run(s);
  }
}

//----------------------------------------------------------------
//  First iteration
//----------------------------------------------------------------
void SwapInOutScheduler::swap_out_first_iter() {
  // In the first iteration, arrays used in a function are always swapped out.
  const int start_idx = func_idx == 0 ? 0 : func_block_ends[func_idx - 1];

  for (int i = start_idx; i < func_block_ends[func_idx]; i++) {
    RecType *r = &order[i];
    if (r->tag == RecTag::CLEAR)
      continue;

    if (context_checker(r->ctx, device_ctx)) {
      if (r->sawptr.use_count() <= 1) {
        // If shared_ptr exists only in said_map (= use_count() == 1),
        // this shared_ptr owns intermideate buffer.
        r->temporary_buffer = true;
        auto p = r->sawptr.lock();
        p->clear();
      } else if (auto p = r->sawptr.lock()) {
        // The array is not cleared yet. Swap it out.
        if (is_not_cleared_yet(p))
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
    } else if (!context_checker(r->ctx, host_ctx)) {
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
void SwapInOutScheduler::sa_callback_recorder(
    SyncedArrayPtr saptr, const SyncedArrayCallbackTag sa_tag,
    const dtypes dtype, const Context &ctx, const bool write_only,
    const bool first_creation, const bool off_recording) {
  if (off_recording) {
    return;
  }

  // check context whether is supported by scheduler or not.
  if (sa_tag != SyncedArrayCallbackTag::CLEAR && // when clear, ctx is dummy.
      !context_checker(ctx, device_ctx) && !context_checker(ctx, host_ctx)) {
    NBLA_ERROR(error_code::type,
               "[SwapInOutScheduler] Unsupported array type: " +
                   ctx.array_class);
  }

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
void SwapInOutScheduler::sa_callback_tracer(
    SyncedArrayPtr saptr, const SyncedArrayCallbackTag sa_tag,
    const dtypes dtype, const Context &ctx, const bool write_only,
    const bool first_creation, const bool off_recording) {
  if (off_recording) {
    return;
  }

  auto tag = convert_tag(sa_tag, write_only);

  // Abort when off-scheduled get/cast happens after preclear destroyed data.
  if (precleared.find(saptr) != precleared.end()) {
    if (tag == RecTag::CLEAR) { // Actual clear. it is Ok.
      precleared.erase(saptr);
    } else { // Unexpected get/cast. Abort.
      NBLA_ERROR(error_code::unclassified,
                 "Off-scheduled get or cast appears after preclear.");
    }
  }

  // Reset the cast prefetch flag when the real cast was called.
  if (cast_prefetched.find(saptr) != cast_prefetched.end()) {
    if (tag == RecTag::CAST) {
      cast_prefetched.erase(saptr);
    }
  }

  // Check if the on-going iteration follows the recorded order.
  auto r = order[order_idx];
  auto rec_saptr = r.sawptr.lock();

  // Case 1: Over the current function block.
  if (order_idx >= func_block_ends[func_idx]) {
#if 0
      std::cerr << "[wrong order] type 1: order idx exceeds recorded one." << std::endl;
#endif
    wrong_ordered.push_back(
        {tag, 0, saptr, saptr->size(), dtype, ctx, false, false});
  }
  // Case 2: SyncedArray replacement in the same order
  else if (r.sawptr.expired() && // "expired" implies the replacement.
           tag == r.tag && saptr->size() == r.size && dtype == r.dtype &&
           ctx.array_class == r.ctx.array_class) {
#if 0
      std::cout << "[wrong order] type 2: SyncedArray would be replaced." << std::endl;
#endif
    // Replace all recorded SyncedArray with new one
    for (auto &i : said_to_order_idx[r.said]) {
      order[i].sawptr = saptr;
    }
  }
  // Case 3: Different entry
  else if (tag != r.tag || saptr != rec_saptr || saptr->size() != r.size ||
           dtype != r.dtype || ctx.array_class != r.ctx.array_class) {
#if 0
      std::cout << "[wrong order] type 3: Different entry." << std::endl;
#endif

    // Abort when off-scheduled get/cast happens after prefetch cast erased
    // the original data type.
    if (cast_prefetched[saptr]) {
      if (!cast_prefetch_no_abort && tag != RecTag::CAST) {
        NBLA_ERROR(error_code::unclassified,
                   "Off-scheduled get or cast of the same array"
                   "appears after cast prefetch.");
      }
    }

    wrong_ordered.push_back(
        {tag, 0, saptr, saptr->size(), dtype, ctx, false, false});
  }

  order_idx++;
}

SwapInOutScheduler::RecTag
SwapInOutScheduler::convert_tag(const SyncedArrayCallbackTag sa_tag,
                                const bool write_only) {
  if (sa_tag == SyncedArrayCallbackTag::GET) {
    return RecTag::GET;
  } else if (sa_tag == SyncedArrayCallbackTag::CAST) {
    return RecTag::CAST;
  } else if (sa_tag == SyncedArrayCallbackTag::CLEAR) {
    return RecTag::CLEAR;
  }

  NBLA_ERROR(error_code::type, "Unsupported SyncedArrayCallbackTag");
}
}
