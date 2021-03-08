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

#include <nbla/singleton_manager-internal.hpp>

namespace nbla {

SingletonManager *SingletonManager::self_ = nullptr;

void SingletonManager::clear() {
  SingletonManager &s = get_self();
  for (int i = 0; i < s.count_; ++i) {
    erase_by_id(i); // Delete singleton if it's active.
  }
  // Clear all
  s.singletons_.clear();
  s.adr2id_.clear();
  s.count_ = 0;
}

void SingletonManager::erase_by_id(int id) {
  SingletonManager &s = get_self();
  auto it = s.singletons_.find(id);
  if (it == s.singletons_.end())
    return;
  it->second.second(); // Call deleter.
  s.adr2id_.erase(it->second.first);
  s.singletons_.erase(it);
}

SingletonManager &SingletonManager::get_self() {
  // TODO: thread-safe
  if (!self_) {
    self_ = new SingletonManager();
  }
  return *self_;
}

SingletonManager::SingletonManager() : count_(0) {}
SingletonManager::~SingletonManager() { clear(); }

///////////////////
// Impl of NNabla
///////////////////
NNabla::NNabla() {}

NNabla::~NNabla() {}

const void *async_get(const shared_ptr<SyncedArray> &arr, dtypes dtype,
                      const Context &ctx) {
  auto ret = arr->get(dtype, ctx, AsyncFlag::ASYNC)->const_pointer<void>();
  arr->get(dtype, ctx); // Workaraound to wait async copy. call get again by the
                        // same dtype and ctx.
  return ret;
}

const void *NNabla::ones(Size_t size, dtypes dtype, const Context &ctx) {
  auto tid = std::this_thread::get_id();
  shared_ptr<SyncedArray> ones;
  std::lock_guard<decltype(mtx_ones_)> lock(mtx_ones_);
  auto it = ones_.find(tid);
  if (it == ones_.end()) {
    ones = std::make_shared<SyncedArray>(size);
    ones->fill(1);
    ones_[tid] = ones;
    return async_get(ones, dtype, ctx);
  }
  ones = it->second;
  if (size > ones->size()) {
    ones = std::make_shared<SyncedArray>(size);
    ones->fill(1);
    ones_[tid] = ones;
  }
  return async_get(ones, dtype, ctx);
}

const void *NNabla::zeros(Size_t size, dtypes dtype, const Context &ctx) {
  auto tid = std::this_thread::get_id();
  shared_ptr<SyncedArray> zeros;
  std::lock_guard<decltype(mtx_zeros_)> lock(mtx_zeros_);
  auto it = zeros_.find(tid);
  if (it == zeros_.end()) {
    zeros = std::make_shared<SyncedArray>(size);
    zeros->zero();
    ones_[tid] = zeros;
    return async_get(zeros, dtype, ctx);
  }
  zeros = it->second;
  if (size > zeros->size()) {
    zeros = std::make_shared<SyncedArray>(size);
    zeros->zero();
    ones_[tid] = zeros;
  }
  return async_get(zeros, dtype, ctx);
}

NBLA_INSTANTIATE_SINGLETON(NBLA_API, NNabla);
}
