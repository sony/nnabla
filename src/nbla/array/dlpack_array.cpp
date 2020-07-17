// Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

#include <nbla/array/dlpack_array.hpp>
#include <nbla/utils/dlpack_utils.hpp>

namespace nbla {

DlpackArray::DlpackArray(const Size_t size, dtypes dtype, const Context &ctx)
    : Array(size, dtype, ctx, AllocatorMemory()) {}

DlpackArray::~DlpackArray() {
  call_deleter(dlp_); // Call the given deleter to return the borrowed array.
}

inline void *get_ptr_with_offset(const DLTensor &dl_tensor) {
  return reinterpret_cast<void *>(reinterpret_cast<char *>(dl_tensor.data) +
                                  dl_tensor.byte_offset);
}

void DlpackArray::borrow(DLManagedTensor *dlp) {
  dlp_ = dlp;
  ptr_ = get_ptr_with_offset(dlp->dl_tensor);
}

void DlpackArray::copy_from(const Array *src_array) {
  NBLA_ERROR(error_code::not_implemented,
             "DlpackArray must implement copy_from(const Array *src_array).");
}

void DlpackArray::zero() {
  NBLA_ERROR(error_code::not_implemented, "Array must implement zero().");
}

void DlpackArray::fill(float value) {
  NBLA_ERROR(error_code::not_implemented,
             "Array must implement fill(float value).");
}

Context DlpackArray::filter_context(const Context &ctx) {
  NBLA_ERROR(error_code::not_implemented,
             "Array must implement filter_context(const Context&).");
}
}
