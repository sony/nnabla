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

#include <nbla/array_registry.hpp>
#include <nbla/context.hpp>
#include <nbla/synced_array.hpp>

namespace nbla {
// Constructor
SyncedArray::SyncedArray(const Size_t size)
    : head_{"", "", dtypes::FLOAT}, zeroing_(false), filling_(false),
      modification_count_(0) {
  size_ = size;
}

SyncedArray::~SyncedArray() {}

Array *SyncedArray::cast(dtypes dtype, const Context &ctx, bool write_only) {
  return cast_sp(dtype, ctx, write_only).get();
}

shared_ptr<Array> SyncedArray::cast_sp(dtypes dtype, const Context &ctx,
                                       bool write_only) {
  // 1. Create an array and/or synchronize with the head.
  head_ = sync(dtype, ctx, write_only); // cast() changes head.
  // 2. Clear all previous arrays.
  auto created_array = array_[head_.key];
  clear_all_array();
  created_array.second = true;
  array_[head_.key] = created_array;
  // 3. Increment modification count to let solver to know whether it's modified
  // or not
  modification_count_++;
  // 4. Return a requested array
  return created_array.first;
}

const Array *SyncedArray::get(dtypes dtype, const Context &ctx) {
  ArrayDesc desc = sync(dtype, ctx); // get() does not change head.
  array_[desc.key].second = true;    // Set as at-head.
  return array_[desc.key].first.get();
}

void SyncedArray::zero() {
  clear_all_array();
  zeroing_ = true;
  modification_count_++;
}

void SyncedArray::fill(float value) {
  clear_all_array();
  filling_ = true;
  fill_value_ = value;
  modification_count_++;
}

size_t SyncedArray::modification_count() const { return modification_count_; }

SyncedArray::ArrayDesc SyncedArray::sync(dtypes dtype, const Context &ctx_orig,
                                         bool write_only) {
  Context ctx = ArrayCreator::filter_context(ctx_orig);
  ArrayDesc desc{get_array_key_from_context(ctx) + ":" + dtype_to_string(dtype),
                 ctx.array_class, dtype};
  // Specified array is not allocated
  if (array_.find(desc.key) == array_.end()) {
    if (array_.size() == 0)
      head_ = desc;
    array_[desc.key] = std::make_pair(
        shared_ptr<Array>(ArrayCreator::create(size_, dtype, ctx)), false);
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
      array->copy_from(head_array);
    } else {
      ArraySynchronizer::synchronize(head_.array_class, head_array,
                                     desc.array_class, array);
    }
  }
  return desc;
}

void SyncedArray::clear() {
  array_.clear();
  zeroing_ = false;
  filling_ = false;
}

// Reset head state
void SyncedArray::clear_all_array() {
  array_.clear();
  zeroing_ = false;
  filling_ = false;
}
}
