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

// array.cpp
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/exception.hpp>

#include <vector>

namespace nbla {

using std::vector;

Array::Array(const Size_t size, dtypes dtype, const Context &ctx,
             AllocatorMemory &&mem)
    : size_(size), dtype_(dtype), ctx_(ctx), mem_(std::move(mem)) {}

Array::~Array() {}

size_t Array::size_as_bytes(Size_t size, dtypes dtype) {
  return size * sizeof_dtype(dtype);
}

Context Array::filter_context(const Context &ctx) {
  NBLA_ERROR(error_code::not_implemented,
             "Array must implement filter_context(const Context&).");
}
}
