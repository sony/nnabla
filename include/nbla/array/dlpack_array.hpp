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

#ifndef __NBLA_DLPACK_ARRAY_HPP__
#define __NBLA_DLPACK_ARRAY_HPP__

#include <dlpack/dlpack.h> // third-party
#include <nbla/array.hpp>

namespace nbla {
/** Array with a borrwed memory pointer from other frameworks via DLPack.
*/
class NBLA_API DlpackArray : public Array {
protected:
  DLManagedTensor *dlp_ = nullptr;
  void *ptr_ = nullptr; // Borrowed memory pointer added "byte_offset".

  // Return the borrowed memory pointer.
  virtual void *mem_pointer() { return ptr_; }
  virtual const void *mem_const_pointer() const { return ptr_; }

public:
  /** Constructor not to finish the construction of this class.

      This special constructor is expected to be called from
      ArrayCreator::create in SyncedArray::get/cast. After the call,
      DlpackArray::borrow must be done to complete the construction.
   */
  DlpackArray(const Size_t size, dtypes dtype, const Context &ctx);
  virtual ~DlpackArray();
  void borrow(DLManagedTensor *dlp);

  // Not implemented methods in this interface class.
  // Deveclopers must implement them in derived classes.
  // However, you can see the definition of these methods in .cpp file.
  // This is becasuse this class is used like dynamic_cast<DlpackArray>
  // and it needs that this class is not abstruct.
  virtual void copy_from(const Array *src_array);
  virtual void zero();
  virtual void fill(float value);
  static Context filter_context(const Context &ctx);
};
}
#endif
