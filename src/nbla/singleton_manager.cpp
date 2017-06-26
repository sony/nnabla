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
  s.singletons_.erase(it);
  s.adr2id_.erase(it->second.first);
}

SingletonManager &SingletonManager::get_self() {
  static SingletonManager s;
  return s;
}

SingletonManager::SingletonManager() : count_(0) {}
SingletonManager::~SingletonManager() { clear(); }

///////////////////
// Impl of NNabla
///////////////////
NNabla::NNabla() : ones_(new SyncedArray(0)), zeros_(new SyncedArray(0)) {

  ones_->fill(1);
  zeros_->zero();
}

NNabla::~NNabla() {}

const void *NNabla::ones(Size_t size, dtypes dtype, const Context &ctx) {
  if (size > ones_->size()) {
    ones_.reset(new SyncedArray(size));
    ones_->fill(1);
  }
  return ones_->get(dtype, ctx)->const_pointer<void>();
}

const void *NNabla::zeros(Size_t size, dtypes dtype, const Context &ctx) {
  if (size > zeros_->size()) {
    zeros_.reset(new SyncedArray(size));
    zeros_->zero();
  }
  return zeros_->get(dtype, ctx)->const_pointer<void>();
}

NBLA_INSTANTIATE_SINGLETON(NBLA_API, NNabla);
}
