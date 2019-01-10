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

#ifndef __NBLA_MEMORY_INTERNAL_HPP__
#define __NBLA_MEMORY_INTERNAL_HPP__
#include <nbla/memory.hpp>

#ifdef NBLA_VERBOSE_MEMORY_USAGE
#include <iostream>
#include <typeinfo>
#endif

namespace nbla {

template <typename M>
shared_ptr<M> MemoryCache<M>::pop_or_create(Size_t bytes,
                                            const string &device) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (pools_.count(device) == 0) {
    // First access the device.
    pools_.insert({device, cache_type()});
  }
  // Get pool of the device.
  cache_type &pool = pools_[device]; // unordered_map<int, CudaMemoryPtr>

  // Allocation size will be a multiple of blocksize_ except scalar (infer
  // from
  // bytes)
  size_t alloc_bytes =
      bytes > 8 ? ((bytes + blocksize_ - 1) / blocksize_) * blocksize_ : bytes;

  // Create a CudaMemory instance. Note that memory is not allocated at this
  // moment.
  auto newmem = make_shared<memory_type>(alloc_bytes, device);
  // Find
  for (int r = 0; r < 2; ++r) {
    auto it = pool.find(newmem->size());
    if (it != pool.end()) {
      // Reuse from cache.
      auto ret = it->second;
      pool.erase(it);
      return ret;
    }

#ifdef NBLA_VERBOSE_MEMORY_USAGE
    // TODO: enable debut print
    std::cout << "MemoryCache<" << typeid(M).name() << ">: Creating with size "
              << alloc_bytes << std::endl;
#endif

    // Allocate a new memory
    if (newmem->allocate())
      return newmem;
  }
  // Delete all cache.
  // TODO: Enable debug print
  // std::cout << "Clearing memory pool." << std::endl;
  pool.clear();
  NBLA_CHECK(newmem->allocate(), error_code::memory,
             "Could not allocate a new memory of size (%s of %s bytes) in %s.",
             std::to_string(newmem->size()).c_str(),
             std::to_string(bytes).c_str(), typeid(M).name());
  return newmem;
}

template <typename M> void MemoryCache<M>::cache(shared_ptr<memory_type> mem) {
  std::lock_guard<std::mutex> lock(mtx_);

  const string &device = mem->device();
  if (pools_.count(device) == 0) {
    // First access the device.
    pools_.insert({device, cache_type()});
  }
  pools_[device].insert({mem->size(), mem});
  // TODO: Enable debug print
  // std::cout << "Caching back (size=" << mem->size() << ") (" <<
  // pools_[device].size() << ")" << std::endl;
}

template <typename M>
size_t MemoryCache<M>::count(const string &device_id) const {
  std::lock_guard<std::mutex> lock(mtx_);

  if (pools_.count(device_id) == 0) {
    return 0;
  }
  return pools_.at(device_id).size();
}

template <typename M> void MemoryCache<M>::clear() {
  std::lock_guard<std::mutex> lock(mtx_);

  for (auto &kv : pools_) {
    kv.second.clear();
  }
}

template <typename M> void MemoryCache<M>::clear(const string &device_id) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (pools_.count(device_id) == 0)
    return;
  pools_[device_id].clear();
}

template <typename M>
MemoryCache<M>::MemoryCache(int blocksize) : blocksize_(blocksize) {}
template <typename M> MemoryCache<M>::~MemoryCache() { pools_.clear(); }
}

#endif
