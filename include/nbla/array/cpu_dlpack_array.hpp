// Copyright 2020,2021 Sony Corporation.
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

#ifndef __NBLA_CPU_DLPACK_ARRAY_HPP__
#define __NBLA_CPU_DLPACK_ARRAY_HPP__

#include <nbla/array/dlpack_array.hpp>

namespace nbla {

/** CPU array with a borrwed memory pointer from other frameworks via DLPack.
*/
class NBLA_API CpuDlpackArray : public DlpackArray {
public:
  explicit CpuDlpackArray(const Size_t size, dtypes dtype, const Context &ctx);
  virtual ~CpuDlpackArray();
  virtual void copy_from(const Array *src_array);
  virtual void zero();
  virtual void fill(float value);
  static Context filter_context(const Context &ctx);
};
}
#endif