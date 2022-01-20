// Copyright 2020,2021 Sony Corporation.
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

#include "./cpu_array-internal.hpp"
#include <nbla/array/cpu_dlpack_array.hpp>

#include <cstring>

namespace nbla {

CpuDlpackArray::CpuDlpackArray(const Size_t size, dtypes dtype,
                               const Context &ctx)
    : DlpackArray(size, dtype, ctx) {}

CpuDlpackArray::~CpuDlpackArray() {}

Context CpuDlpackArray::filter_context(const Context &ctx) {
  return Context({}, "CpuDlpackArray", "");
}

void CpuDlpackArray::zero() {
  std::memset(this->pointer<void>(), 0,
              this->size() * sizeof_dtype(this->dtype_));
}

NBLA_DEFINE_FUNC_COPY_FROM(CpuDlpackArray, cpu_array_copy, cpu);
NBLA_DEFINE_FUNC_FILL(CpuDlpackArray, cpu_fill, cpu);
}
