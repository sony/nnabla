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

#ifndef __NBLA_CPU_ARRAY_HPP__
#define __NBLA_CPU_ARRAY_HPP__

#include <nbla/array.hpp>
#include <nbla/memory/allocator.hpp>

namespace nbla {

/** Array for CPU.
\ingroup ArrayImplGrp
*/
class NBLA_API CpuArray : public Array {
protected:
public:
  CpuArray(const Size_t size, dtypes dtype, const Context &ctx);
  CpuArray(const Size_t size, dtypes dtype, const Context &ctx,
           AllocatorMemory &&mem);
  virtual ~CpuArray();
  virtual void copy_from(const Array *src_array);
  virtual void zero();
  virtual void fill(float value);
  static Context filter_context(const Context &ctx);
};

/** Cached CPU array.
 */
class NBLA_API CpuCachedArray : public CpuArray {
public:
  explicit CpuCachedArray(const Size_t size, dtypes dtype, const Context &ctx);
  virtual ~CpuCachedArray();
  static Context filter_context(const Context &ctx);
};
}
#endif
