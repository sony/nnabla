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

Array *SyncedArray::cast(dtypes dtype, const Context &ctx) {
  head_ = sync(dtype, ctx); // cast() changes head.
  reset_head(); // This will clear all head state, zeroing and filling flags.
  array_[head_.key].second = true; // Set as at-head.
  modification_count_++;
  return array_[head_.key].first.get();
}

const Array *SyncedArray::get(dtypes dtype, const Context &ctx) {
  ArrayDesc desc = sync(dtype, ctx); // get() does not change head.
  array_[desc.key].second = true;    // Set as at-head.
  return array_[desc.key].first.get();
}

void SyncedArray::zero() {
  reset_head();
  zeroing_ = true;
  modification_count_++;
}

void SyncedArray::fill(float value) {
  reset_head();
  filling_ = true;
  fill_value_ = value;
  modification_count_++;
}

size_t SyncedArray::modification_count() const { return modification_count_; }

SyncedArray::ArrayDesc SyncedArray::sync(dtypes dtype,
                                         const Context &ctx_orig) {
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
  auto ah = array_[desc.key];
  Array *array = ah.first.get();
  bool at_head = ah.second;

  // Not initialized or the array is not at head.
  if (!at_head) {
    if (zeroing_) {
      // Do lazy evaluation of zero().
      array->zero();
    } else if (filling_) {
      // Do lazy evaluation of fill().
      array->fill(fill_value_);
    } else if (array_.size() > 1) {
      Array *head_array = array_[head_.key].first.get();
      if (head_.array_class == desc.array_class) {
        array->copy_from(head_array);
      } else {
        ArraySynchronizer::synchronize(head_.array_class, head_array,
                                       desc.array_class, array);
      }
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
void SyncedArray::reset_head() {
  for (auto &kv : array_) {
    kv.second.second = false;
  }
  zeroing_ = false;
  filling_ = false;
}
}
