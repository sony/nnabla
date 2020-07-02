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

// cpu_array.cpp
#include <nbla/array/cpu_array.hpp>
#include <nbla/array_registry.hpp>
#include <nbla/common.hpp>
#include <nbla/cpu.hpp>

#include <cstring> // memset
#include <vector>

namespace nbla {

using std::vector;
using std::shared_ptr;
using std::make_shared;

CpuArray::CpuArray(const Size_t size, dtypes dtype, const Context &ctx)
    : Array::Array(size, dtype, ctx,
                   SingletonManager::get<Cpu>()->naive_allocator()->alloc(
                       Array::size_as_bytes(size, dtype), "")) {}

CpuArray::CpuArray(const Size_t size, dtypes dtype, const Context &ctx,
                   AllocatorMemory &&mem)
    : Array::Array(size, dtype, ctx, std::move(mem)) {}

CpuArray::~CpuArray() {}

void CpuArray::zero() {
  std::memset(this->pointer<void>(), 0,
              this->size() * sizeof_dtype(this->dtype_));
}

/** Helper template to copy data from CpuArray with other data type.
*/
template <typename Ta, typename Tb>
void cpu_array_copy(const Array *src, Array *dst) {
  const Ta *p_src = src->const_pointer<Ta>();
  Tb *p_dst = dst->pointer<Tb>();
  if (!src->size()) {
    // zero-size means scalar
    *p_dst = *p_src;
    return;
  }
  std::copy(p_src, p_src + src->size(), p_dst);
}

template <typename T> void cpu_fill(Array *self, float value) {
  T *ptr = self->pointer<T>();
  size_t size = self->size();
  std::fill(ptr, ptr + size, static_cast<T>(value));
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
                               const Context &ctx)
    : CpuArray(size, dtype, ctx,
               SingletonManager::get<Cpu>()->caching_allocator()->alloc(
                   Array::size_as_bytes(size, dtype), "")) {}

CpuCachedArray::~CpuCachedArray() {}

Context CpuCachedArray::filter_context(const Context &ctx) {
  return Context({}, "CpuCachedArray", "");
}
}
