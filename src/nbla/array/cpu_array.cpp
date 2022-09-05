// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

// cpu_array.cpp
#include "./cpu_array-internal.hpp"
#include <nbla/array_registry.hpp>
#include <nbla/common.hpp>
#include <nbla/cpu.hpp>

#include <cstring> // memset
#include <vector>

namespace nbla {

CpuArray::CpuArray(const Size_t size, dtypes dtype, const Context &ctx,
                   const AllocatorMemoryPtr mem, const Size_t offset)
    : Array::Array(size, dtype, ctx,
                   mem ? mem
                       : SingletonManager::get<Cpu>()->naive_allocator()->alloc(
                             Array::size_as_bytes(size, dtype), ""),
                   offset) {}

CpuArray::~CpuArray() {}

void CpuArray::zero() {
  std::memset(this->pointer<void>(), 0,
              this->size() * sizeof_dtype(this->dtype_));
}

Context CpuArray::filter_context(const Context &ctx) {
  return Context({}, "CpuArray", "");
}

NBLA_DEFINE_FUNC_COPY_FROM(CpuArray, cpu_array_copy, cpu);
NBLA_DEFINE_FUNC_FILL(CpuArray, cpu_fill, cpu);

/////////////////////////////////
// CpuCachedArray implementation
/////////////////////////////////
CpuCachedArray::CpuCachedArray(const Size_t size, dtypes dtype,
                               const Context &ctx, const AllocatorMemoryPtr mem,
                               const Size_t offset)
    : CpuArray(size, dtype, ctx,
               mem ? mem
                   : SingletonManager::get<Cpu>()->caching_allocator()->alloc(
                         Array::size_as_bytes(size, dtype), ""),
               offset) {}

CpuCachedArray::~CpuCachedArray() {}

Context CpuCachedArray::filter_context(const Context &ctx) {
  return Context({}, "CpuCachedArray", "");
}
}
