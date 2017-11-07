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
#include <nbla/cpu_memory.hpp>

#include <cstring> // memset
#include <vector>

namespace nbla {

using std::vector;
using std::shared_ptr;
using std::make_shared;

CpuArray::CpuArray(const Size_t size, dtypes dtype, const Context &ctx)
    : Array::Array(size, dtype, ctx), inuse_memory_(nullptr) {}

CpuArray::~CpuArray() {
  if (this->object_) {
    this->deallocate();
  }
}

void CpuArray::zero() {
  std::memset(this->pointer<void>(), 0,
              this->size() * sizeof_dtype(this->dtype_));
}

void CpuArray::allocate() {
#ifdef NBLA_VERBOSE_MEMORY_USAGE
  printf("CpuArray is created with size of %d\n",
         (int)(this->size_ * sizeof(this->dtype_)));
#endif
  int msize = this->size_ * sizeof_dtype(this->dtype_);
  inuse_memory_ = make_shared<CpuMemory>(msize, "");
  inuse_memory_->allocate();
  this->object_ = inuse_memory_->ptr();
}
void CpuArray::deallocate() {
  inuse_memory_ = nullptr;
  this->object_ = nullptr;
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

NBLA_DEFINE_FUNC_COPY_FROM(CpuArray, cpu_array_copy);

template <typename T> void cpu_fill(Array *self, float value) {
  T *ptr = self->pointer<T>();
  size_t size = self->size();
  std::fill(ptr, ptr + size, static_cast<T>(value));
}

NBLA_DEFINE_FUNC_FILL(CpuArray, cpu_fill);

Context CpuArray::filter_context(const Context &ctx) {
  return Context("", "CpuArray", "", "");
}

/////////////////////////////////
// CpuCachedArray implementation
/////////////////////////////////
CpuCachedArray::CpuCachedArray(const Size_t size, dtypes dtype,
                               const Context &ctx)
    : CpuArray(size, dtype, ctx) {}

CpuCachedArray::~CpuCachedArray() { this->deallocate(); }

void CpuCachedArray::allocate() {
  deallocate();
  int bytes = this->size_ * sizeof_dtype(this->dtype_);
  auto mem = SingletonManager::get<Cpu>()->memcache().pop_or_create(bytes, "");
  this->object_ = mem->ptr();
  this->inuse_memory_ = mem;
}

void CpuCachedArray::deallocate() {
  if (this->inuse_memory_) {
    SingletonManager::get<Cpu>()->memcache().cache(this->inuse_memory_);
    this->inuse_memory_ = nullptr;
  }
}
Context CpuCachedArray::filter_context(const Context &ctx) {
  return Context("", "CpuCachedArray", "", "");
}
}
