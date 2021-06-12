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

#ifndef __NBLA_DLPACK_UTILS_HPP__
#define __NBLA_DLPACK_UTILS_HPP__

#include <dlpack/dlpack.h> // included from the third-party directory
#include <nbla/nd_array.hpp>

namespace nbla {

/** NNabla borrows a DLPack Tensor as a NdArray.

    NdArray is newly created shared the memory in DLPack and is returned.
*/
NBLA_API NdArrayPtr from_dlpack(DLManagedTensor *from);

/** NNabla borrows a DLPack Tensor as a NdArray.

    The passed NdArray is destroyed by this function. Into the NdArray,
    a array shared the memory in DLPack is newly set.
 */
NBLA_API void from_dlpack(DLManagedTensor *from, NdArray *to);

/** NNabla lends the head array in a NdArray as a DLPack Tensor.
*/
NBLA_API DLManagedTensor *to_dlpack(NdArray *array);

/** NNabla lends a array in NdArray as a DLPack Tensor
    according to given dtype and context.
 */
NBLA_API DLManagedTensor *to_dlpack(NdArray *array, const dtypes dtype,
                                    const Context &ctx);

/** Call deleter in DLManagedTensor and then delete DLManagedTensor itself
 */
NBLA_API void call_deleter(DLManagedTensor *dlp);

/** Convert DLDataType to Nnabla dtype
*/
dtypes convert_dlpack_type_to_dtype(const DLDataType &dlp_type);
}
#endif
