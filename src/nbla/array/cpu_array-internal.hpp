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

#ifndef NBLA_CPU_ARRAY_INTERNAL_HPP_
#define NBLA_CPU_ARRAY_INTERNAL_HPP_
#include <nbla/array/cpu_array.hpp>

namespace nbla {
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

NBLA_DEFINE_COPY_WRAPPER(cpu_array_copy);
} // End of namespace nbla
#endif
