// Copyright (c) 2017-2020 Sony Corporation. All Rights Reserved.
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

#include <nbla/array_registry.hpp>
#include <nbla/context.hpp>
#include <nbla/singleton_manager-internal.hpp>
#include <nbla/synced_array.hpp>

#ifdef ENABLE_SYNC_DEBUG
#include <cstdlib>
static bool sync_debug_enabled() {
  static bool enabled = false;
  if (enabled) {
    return enabled;
  }
  const char *env_c = std::getenv("NNABLA_SYNC_DEBUG");
  if (env_c == nullptr) {
    return true;
  }
  std::string env = std::string(env_c);
  try {
    if (std::stoi(env) == 0) {
      return false;
    }
  } catch (...) {
  }
  enabled = true;
  return true;
}

#define SYNC_DEBUG(...)                                                        \
  if (sync_debug_enabled()) {                                                  \
    printf(__VA_ARGS__);                                                       \
    printf("\n");                                                              \
  }
#else
#define SYNC_DEBUG(...)
#endif

namespace nbla {
// Constructor
SyncedArray::SyncedArray(const Size_t size)
    : head_{"", "", dtypes::FLOAT}, zeroing_(false), filling_(false),
      modification_count_(0), clear_called_(false) {
  size_ = size;
}

SyncedArray::~SyncedArray() {}

Array *SyncedArray::cast(dtypes dtype, const Context &ctx, bool write_only,
                         const int async_flags) {
  return cast_sp(dtype, ctx, write_only, async_flags).get();
}

shared_ptr<Array> SyncedArray::cast_sp(dtypes dtype, const Context &ctx,
                                       bool write_only, const int async_flags) {
  // This array is created at first time.
  const bool first_creation = (get_num_arrays() == 0);

  // 1. Create an array and/or synchronize with the head.
  head_ = sync(dtype, ctx, write_only, async_flags); // cast() changes head.
  // 2. Clear all previous arrays.
  auto created_array = array_[head_.key];
  clear_all_array();
  created_array.second = true;
  array_[head_.key] = created_array;
  // 3. Increment modification count to let solver to know whether it's modified
  // or not
  modification_count_++;
  clear_called_ = false;

  // 4. Call a callback function
  const bool off_recording = (bool)(async_flags & AsyncFlag::OFFREC);
  SingletonManager::get<SyncedArrayCallback>()->call_callback(
      shared_from_this(), SyncedArrayCallbackTag::CAST,
      created_array.first->dtype(), ctx, write_only, first_creation,
      off_recording);

  // 5. Return a requested array
  return created_array.first;
}

const Array *SyncedArray::get(dtypes dtype, const Context &ctx,
                              const int async_flags) {
  return get_sp(dtype, ctx, async_flags).get();
}

shared_ptr<const Array> SyncedArray::get_sp(dtypes dtype, const Context &ctx,
                                            const int async_flags) {
  // This array is created at first time.
  const bool first_creation = (get_num_arrays() == 0);

  clear_called_ = false;

  ArrayDesc desc =
      sync(dtype, ctx, false, async_flags); // get() does not change head.
  array_[desc.key].second = true;           // Set as at-head.

  // Call a callback function
  const bool off_recording = (bool)(async_flags & AsyncFlag::OFFREC);
  SingletonManager::get<SyncedArrayCallback>()->call_callback(
      shared_from_this(), SyncedArrayCallbackTag::GET,
      array_[desc.key].first->dtype(), ctx, false, first_creation,
      off_recording);

  return std::const_pointer_cast<const Array>(array_[desc.key].first);
}

Array *SyncedArray::head_array() { return head_array_sp().get(); }

shared_ptr<Array> SyncedArray::head_array_sp() {
  return array_[head_.key].first;
}

const void *SyncedArray::data_ptr(dtypes dtype, const Context &ctx,
                                  bool write_only, const int async_flags) {
  cast_sp(dtype, ctx, write_only, async_flags);
  return array_[head_.key].first->const_pointer();
}

void SyncedArray::zero() {
  clear();
  zeroing_ = true;
  modification_count_++;
  clear_called_ = false;
}

void SyncedArray::fill(float value) {
  clear();
  filling_ = true;
  fill_value_ = value;
  modification_count_++;
  clear_called_ = false;
}

size_t SyncedArray::modification_count() const { return modification_count_; }

bool SyncedArray::clear_called() const { return clear_called_; }

inline string create_key(const dtypes &dtype, const Context &ctx) {
  // Array classes in the same array group are identified in SyncedArray.
  return ctx.device_id + ":" + ArrayGroup::get_group(ctx.array_class) + ":" +
         dtype_to_string(dtype);
}

SyncedArray::ArrayDesc SyncedArray::sync(dtypes dtype, const Context &ctx_orig,
                                         bool write_only,
                                         const int async_flags) {
  Context ctx = ArrayCreator::filter_context(ctx_orig);
  ArrayDesc desc{create_key(dtype, ctx), ctx.array_class, dtype};

  // Specified array is not allocated
  if (array_.find(desc.key) == array_.end()) {
    if (array_.size() == 0)
      head_ = desc;
    array_[desc.key] = std::make_pair(
        shared_ptr<Array>(ArrayCreator::create(size_, dtype, ctx)), false);
  } else {
    // Wait for the end of previous async_flags asynchronous memcpy
    array_[desc.key].first->wait_event(ctx, async_flags);
  }

  if (write_only) {
    return desc;
  }

  auto ah = array_[desc.key];
  Array *array = ah.first.get();
  bool at_head = ah.second;
  // Not initialized or the array is not at head.
  if (at_head) {
    return desc;
  }
  if (zeroing_) {
    // Do lazy evaluation of zero().
    array->zero();
  } else if (filling_) {
    // Do lazy evaluation of fill().
    array->fill(fill_value_);
  } else if (array_.size() > 1) {
    // TODO: Better heuristic choice from current heads
    Array *head_array = array_[head_.key].first.get();
    if (head_.array_class == desc.array_class) {
      head_array->wait_event(ctx, async_flags);
      array->copy_from(head_array);
    } else {
      ArraySynchronizer::synchronize(head_.array_class, head_array,
                                     desc.array_class, array, async_flags);
      SYNC_DEBUG("SYNC: %s<%s> --[%ld elements (%ld bytes in %s)]--> %s<%s>.",
                 head_.array_class.c_str(),
                 dtype_to_string(head_.dtype).c_str(),
                 size() / sizeof_dtype(head_.dtype), size(),
                 dtype_to_string(head_.dtype).c_str(), desc.array_class.c_str(),
                 dtype_to_string(desc.dtype).c_str());
    }
  }
  return desc;
}

void SyncedArray::copy_from(const SyncedArray *src) {
  NBLA_CHECK(!src->head_.key.empty(), error_code::value,
             "Source doesn't have any array.");
  auto src_array = src->array_.at(src->head_.key).first;
  auto ctx = src_array->context();
  auto dtype = src_array->dtype();
  auto dst_array = this->cast(dtype, ctx, true);
  dst_array->copy_from(src_array.get());
}

// Public clear calls a callback function.
void SyncedArray::clear() {
  this->clear_all_array();

  // Call a callback function
  SingletonManager::get<SyncedArrayCallback>()->call_callback(
      shared_from_this(), SyncedArrayCallbackTag::CLEAR,
      dtypes::BYTE, // dummy
      Context({"dummy"}, "dummy", "dummy"), false, false, false);
}

// Reset head state. Private clear does not call a callback function.
void SyncedArray::clear_all_array() {
  array_.clear();
  this->clear_flags();
  modification_count_ = 0;
  clear_called_ = true;
}

void SyncedArray::clear_flags() {
  zeroing_ = false;
  filling_ = false;
}

bool SyncedArray::zeroing() const { return zeroing_; }

// Callback function
SyncedArrayCallback::SyncedArrayCallback() : callback_func_(nullptr) {}

SyncedArrayCallback::~SyncedArrayCallback() {}

bool SyncedArrayCallback::empty() { return callback_func_ == nullptr; }

void SyncedArrayCallback::set_callback_func(synced_array_callback_func_type f) {
  callback_func_ = f;
}

void SyncedArrayCallback::call_callback(SyncedArrayPtr saptr,
                                        const SyncedArrayCallbackTag func_name,
                                        const dtypes dtype, const Context &ctx,
                                        const bool write_only,
                                        const bool first_creation,
                                        const bool off_recording) {
  if (!empty()) {
    callback_func_(saptr, func_name, dtype, ctx, write_only, first_creation,
                   off_recording);
  }
}

NBLA_INSTANTIATE_SINGLETON(NBLA_API, SyncedArrayCallback);
}
