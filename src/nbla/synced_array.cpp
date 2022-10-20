// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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

#include <algorithm>

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
  string env = string(env_c);
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
// Array classes in the same array group are identified in SyncedArray.
inline string create_key(const dtypes &dtype, const Context &ctx) {
  return ctx.device_id + ":" + ArrayGroup::get_group(ctx.array_class) + ":" +
         dtype_to_string(dtype);
}

inline string create_key(const string &parent_key, const Size_t offset,
                         const Size_t size) {
  return parent_key + ":" + to_string(offset) + ":" + to_string(size);
}

// Constructor
SyncedArray::SyncedArray(const Size_t size)
    : head_{"", "", dtypes::FLOAT}, zeroing_lazy_eval_(true),
      filling_lazy_eval_(false), zeroing_(false), size_(size), offset_(0),
      modification_count_(0), clear_called_(false), parent_(nullptr) {}

SyncedArray::SyncedArray(SyncedArrayPtr parent, const Size_t size,
                         const Size_t offset)
    : head_{"", "", dtypes::FLOAT}, zeroing_lazy_eval_(false),
      filling_lazy_eval_(false), zeroing_(false), size_(size), offset_(offset),
      modification_count_(0), clear_called_(false), parent_(parent) {

  auto begin = offset_;
  auto end = begin + size_ - 1;

  for (auto child : parent_->children_) {
    if (auto c = child.lock()) {
      auto c_begin = c->offset_;
      auto c_end = c_begin + c->size_ - 1;

      NBLA_CHECK(c_end < begin || end < c_begin, error_code::not_implemented,
                 "Currently, it is not possible to create Child-Array(narrowed "
                 "Array) that has overlapped areas.");
    }
  }

  // NOTE: If parent-SyncedArray has called sync() at least once,
  // ArrayCreator::create() is not lazily evaluated. The reason for not creating
  // it on lazy evaluation is that SyncedArray::dtype() depends on head_, so it
  // must be updated to keep up with changes in the parent. If only head_ is
  // updated first, there will be no Array in array_ even though head_ exists.
  if (parent->has_head_array()) {
    create_array_from_parent();
  }
}

SyncedArray::~SyncedArray() {
  if (is_child()) {
    parent_->remove_child(this);
    parent_ = nullptr;
  }
}

void SyncedArray::create_array_from_parent() {
  ArrayDesc desc(parent_->head_);
  desc.key = create_key(desc.key, offset_, size_);
  head_ = desc;

  array_[desc.key] = std::make_pair(
      shared_ptr<Array>(ArrayCreator::create(
          size_, desc.dtype, parent_->head_array()->context(),
          parent_->head_array_sp()->memory(), (parent_->offset() + offset_))),
      false);
}

void SyncedArray::create_array_descendants() {
  for (auto child : children_) {
    if (auto c = child.lock()) {
      if (!c->has_head_array()) {
        c->create_array_from_parent();
        c->create_array_descendants();
      }
    }
  }
}

SyncedArrayPtr SyncedArray::narrow(const Size_t narrow_size,
                                   const Size_t offset) {
  auto child =
      make_shared<SyncedArray>(shared_from_this(), narrow_size, offset);

  children_.emplace_back(weak_ptr<SyncedArray>(child));
  return child;
}

Array *SyncedArray::cast(dtypes dtype, const Context &ctx, bool write_only,
                         const int async_flags) {
  return cast_sp(dtype, ctx, write_only, async_flags).get();
}

shared_ptr<Array> SyncedArray::cast_sp(dtypes dtype, const Context &ctx,
                                       bool write_only, const int async_flags) {
  if (is_child()) {
    auto root_key = get_root()->head_.key;
    auto filtered_ctx = ArrayCreator::filter_context(ctx);
    auto created_key = create_key(dtype, filtered_ctx);

    // If root has never been called sync(), it is created with the current
    // context by sync().
    // So it is not cast of different type and is not an error.
    NBLA_CHECK(
        root_key == created_key || !get_root()->has_head_array(),
        error_code::runtime,
        "cast of child-arrays is not permitted ( '%s' cannot convert to %s)",
        root_key.c_str(), created_key.c_str());
  }

  // This array is created at first time.
  const bool first_creation = (get_num_arrays() == 0);

  // 1. Create an array and/or synchronize with the head.
  auto prev_key = head_.key;
  head_ = sync(dtype, ctx, write_only, async_flags); // cast() changes head.

  // 2. Clear all previous arrays.
  if (has_family()) {
    // Re-create all child-array since different type cast has been called for
    // root-array.
    if ((head_.key != prev_key) && is_root()) {
      clear_all_array_descendants(false);
      create_array_descendants();
    } else {
      auto root = get_root();
      auto head_array = root->array_[root->head_.key];
      root->array_.clear();
      root->array_[root->head_.key] = head_array;
      root->clear_all_array_descendants(true);
    }
  }

  auto created_array = array_[head_.key];
  clear_all_array();
  created_array.second = true;
  array_[head_.key] = created_array;

  // 3. Increment modification count to let solver to know whether it's modified
  // or not
  modification_count_++;
  clear_called_ = false;
  propagate_zeroing_flag(false);

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
  if (is_root()) {
    clear();
    clear_all_array_descendants(false);
  } else {
    clear_all_array_descendants(true);

    // Clear except head_ array of its own direct lineage array.
    if (has_head_array()) {
      for (auto p = shared_from_this(); p; p = p->parent_) {
        auto p_head_array = p->array_[p->head_.key];
        p->array_.clear();
        p->array_[p->head_.key] = p_head_array;
      }
    }
  }
  clear_flags();
  clear_flags_descendants();

  // Set zeroing_lazy_eval_ to true since it is this SyncedArray that executes
  // Array::zero(). When SyncedArray::zeroing() of the child SyncedArray is
  // called, child return true without executing Array::zero(), so propagate
  // only zeroing_ is true.
  zeroing_lazy_eval_ = true;
  zeroing_ = true;
  propagate_zeroing_flag_descendants(true);
  modification_count_++;
  clear_called_ = false;
}

void SyncedArray::fill(float value) {
  if (is_root()) {
    clear();
    clear_all_array_descendants(false);
  } else {
    clear_all_array_descendants(true);

    // Clear except head_ array of its own direct lineage array.
    if (has_head_array()) {
      for (auto p = shared_from_this(); p; p = p->parent_) {
        auto p_head_array = p->array_[p->head_.key];
        p->array_.clear();
        p->array_[p->head_.key] = p_head_array;
      }
    }
  }

  clear_flags();
  clear_flags_descendants();
  propagate_zeroing_flag(false);
  filling_lazy_eval_ = true;
  fill_value_ = value;
  modification_count_++;
  clear_called_ = false;
}

size_t SyncedArray::modification_count() const { return modification_count_; }

bool SyncedArray::clear_called() const { return clear_called_; }

SyncedArray::ArrayDesc SyncedArray::sync(dtypes dtype, const Context &ctx_orig,
                                         bool write_only,
                                         const int async_flags) {
  Context ctx = ArrayCreator::filter_context(ctx_orig);

  // If child-SyncedArray is created while the parent has head array, Array is
  // created in constructor.
  // If sync() is called for child SyncedArray while the parent has never been
  // called sync(), root SyncedArrays is created first.
  if (is_child() && !has_head_array()) {
    get_root()->sync(dtype, ctx_orig, write_only, async_flags);
  }

  ArrayDesc desc{create_key(dtype, ctx), ctx.array_class, dtype};
  // In the case of child array and same dtype, updated key.
  if (is_child() && (get_root()->head_.key == desc.key)) {
    desc.key = create_key(parent_->head_.key, offset_, size_);
  }

  // Specified array is not allocated
  if (array_.find(desc.key) == array_.end()) {
    if (array_.size() == 0)
      head_ = desc;
    array_[desc.key] = std::make_pair(
        shared_ptr<Array>(ArrayCreator::create(size_, dtype, ctx)), false);
    if (is_root() && has_family()) {
      // Child array is always followed by parent's head_, so create here.
      create_array_descendants();
    }
  } else {
    // Wait for the end of previous async_flags asynchronous memcpy
    array_[desc.key].first->wait_event(ctx, async_flags);
  }

  if (has_family() && check_zeroing_filling()) {
    // Do lazy evaluation of zero() or fill().

    SyncedArrayPtr zero_fill_root = shared_from_this();
    for (auto p = parent_; p; p = p->parent_) {
      if (p->filling_lazy_eval_ || p->zeroing_lazy_eval_) {
        zero_fill_root = p;
      }
    }

    zero_fill_root->traverse_zero_fill();
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

  if (!has_family() && zeroing_lazy_eval_) {
    array->zero();
    zeroing_lazy_eval_ = false;
  } else if (!has_family() && filling_lazy_eval_) {
    // Do lazy evaluation of fill().
    array->fill(fill_value_);
    filling_lazy_eval_ = false;
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
                 dtype_to_string(head_.dtype).c_str(), this->size(),
                 this->size() * sizeof_dtype(head_.dtype),
                 dtype_to_string(head_.dtype).c_str(), desc.array_class.c_str(),
                 dtype_to_string(desc.dtype).c_str());
    }
  }
  return desc;
}

SyncedArrayPtr SyncedArray::get_root() {
  if (is_root()) {
    return shared_from_this();
  }
  return parent_->get_root();
}

void SyncedArray::traverse_zero_fill() {
  // Ordering is guaranteed for zero() and fill().
  // Even if Array::zero( or fill) is called here, it is necessary to trace back
  // to the end children because it may be partially overwritten by descendants.
  if (has_head_array()) {
    if (zeroing_lazy_eval_)
      head_array()->zero();
    else if (filling_lazy_eval_)
      head_array()->fill(fill_value_);
    clear_flags();
  }
  for (auto child : children_) {
    if (auto c = child.lock()) {
      c->traverse_zero_fill();
    }
  }
}

bool SyncedArray::check_zeroing_filling_descendants() {
  for (auto child : children_) {
    if (auto c = child.lock()) {
      if (c->zeroing_lazy_eval_ || c->filling_lazy_eval_) {
        return true;
      }
      if (c->check_zeroing_filling_descendants()) {
        return true;
      }
    }
  }
  return false;
}

bool SyncedArray::check_zeroing_filling() {
  if (zeroing_lazy_eval_ || filling_lazy_eval_) {
    return true;
  }

  if (check_zeroing_filling_descendants()) {
    return true;
  }

  // Check the flag of its own direct lineage array.
  for (auto p = parent_; p; p = p->parent_) {
    if (p->zeroing_lazy_eval_ || p->filling_lazy_eval_) {
      return true;
    }
  }
  return false;
}

void SyncedArray::clear_flags_descendants() {
  for (auto child : children_) {
    if (auto c = child.lock()) {
      c->clear_flags();
      c->clear_flags_descendants();
    }
  }
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
  if (is_child()) {
    NBLA_ERROR(error_code::runtime, "clear of child-arrays is not permitted");
  } else {
    this->clear_all_array();
  }

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

void SyncedArray::clear_all_array_descendants(bool keep_head) {
  for (auto child : children_) {
    if (auto c = child.lock()) {
      if (!c->has_head_array()) {
        return;
      }

      if (keep_head) {
        auto head_array = c->array_[c->head_.key];
        c->array_.clear();
        c->array_[c->head_.key] = head_array;
      } else {
        c->clear_all_array();
      }

      c->clear_all_array_descendants(keep_head);
    }
  }
}

void SyncedArray::clear_flags() {
  zeroing_lazy_eval_ = false;
  filling_lazy_eval_ = false;
}

bool SyncedArray::zeroing() const { return zeroing_; }

void SyncedArray::propagate_zeroing_flag_descendants(bool flag) {
  for (auto child : children_) {
    if (auto c = child.lock()) {
      c->zeroing_ = flag;
      c->propagate_zeroing_flag_descendants(flag);
    }
  }
}

void SyncedArray::propagate_zeroing_flag(bool flag) {
  // for child array
  propagate_zeroing_flag_descendants(flag);

  zeroing_ = flag;

  // Update only the flag of its own direct lineage array.
  for (auto p = parent_; p; p = p->parent_)
    p->zeroing_ = flag;
}

void SyncedArray::remove_child(const SyncedArray *child) {
  children_.erase(std::remove_if(children_.begin(), children_.end(),
                                 [child](weak_ptr<SyncedArray> &ptr) {
                                   if (auto c = ptr.lock()) {
                                     return c.get() == child;
                                   }

                                   // expired
                                   return true;
                                 }),
                  children_.end());
}

int SyncedArray::get_python_user_reference_counts() const {
  if (is_root()) {
    return python_user_reference_counts;
  }
  return parent_->get_python_user_reference_counts();
}

void SyncedArray::update_python_user_reference_counts(const int diff) {
  python_user_reference_counts += diff;
  if (is_child()) {
    parent_->update_python_user_reference_counts(diff);
  }
}

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
