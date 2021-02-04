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
#include <nbla/utils/dlpack_array_registry.hpp>
#include <nbla/utils/dlpack_utils.hpp>

namespace nbla {

inline uint8_t convert_dtype_to_dlpack_code(const dtypes dtype) {
// Local macro (undef locally)
#define SET_CODE(DTYPE, DLCODE)                                                \
  case dtypes::DTYPE:                                                          \
    code = DLCODE;                                                             \
    break;
  // End of macro

  uint8_t code;
  switch (dtype) {
    SET_CODE(BOOL, kDLInt);
    SET_CODE(BYTE, kDLInt);
    SET_CODE(SHORT, kDLInt);
    SET_CODE(INT, kDLInt);
    SET_CODE(LONG, kDLInt);
    SET_CODE(LONGLONG, kDLInt);
    SET_CODE(UBYTE, kDLUInt);
    SET_CODE(USHORT, kDLUInt);
    SET_CODE(UINT, kDLUInt);
    SET_CODE(ULONG, kDLUInt);
    SET_CODE(ULONGLONG, kDLUInt);
    SET_CODE(FLOAT, kDLFloat);
    SET_CODE(DOUBLE, kDLFloat);
    SET_CODE(LONGDOUBLE, kDLFloat);
    SET_CODE(HALF, kDLBfloat);
  default:
    NBLA_ERROR(error_code::type, "dtype %s cannot be converted to "
                                 "DLDataTypeCode.",
               dtype_to_string(dtype).c_str());
    break;
  }
  return code;

#undef SET_CODE
}

Shape_t get_shape_with_contiguous_memory(DLManagedTensor *dlp) {
  const auto ndim = dlp->dl_tensor.ndim;
  const auto shape = dlp->dl_tensor.shape;
  const auto strides = dlp->dl_tensor.strides;
  Shape_t ret_shape(ndim);
  Size_t contig_stride = 1;

  for (int i = ndim - 1; i >= 0; i--) {
    NBLA_CHECK(strides[i] == contig_stride, error_code::value,
               "The array elements must be contiguous in memory for NNabla. "
               "Check strides in DLPack DLTensor.");
    contig_stride *= shape[i];
    ret_shape[i] = shape[i];
  }

  return ret_shape;
}

inline bool cmp_bits_dtype(const uint8_t bits, const dtypes dtype) {
  return bits == sizeof_dtype(dtype) * 8;
}

dtypes convert_dlpack_type_to_dtype(const DLDataType &dlp_type) {
  const auto code = dlp_type.code;
  const auto bits = dlp_type.bits;

  NBLA_CHECK(dlp_type.lanes == 1, error_code::value,
             "NNabla does not have vectrized types.");

  // When some dtypes have the same bit size, for example long and long long
  // have 32 bits in some architectures, the order of following "if" statements
  // determines which type is chosen.
  if (code == kDLBfloat && cmp_bits_dtype(bits, dtypes::HALF)) {
    return dtypes::HALF;
  } else if (code == kDLFloat) {
    if (cmp_bits_dtype(bits, dtypes::FLOAT)) {
      return dtypes::FLOAT;
    } else if (cmp_bits_dtype(bits, dtypes::DOUBLE)) {
      return dtypes::DOUBLE;
    } else if (cmp_bits_dtype(bits, dtypes::LONGDOUBLE)) {
      return dtypes::LONGDOUBLE;
    }
  } else if (code == kDLInt) {
    if (cmp_bits_dtype(bits, dtypes::INT)) {
      return dtypes::INT;
    } else if (cmp_bits_dtype(bits, dtypes::BYTE)) {
      return dtypes::BYTE;
    } else if (cmp_bits_dtype(bits, dtypes::SHORT)) {
      return dtypes::SHORT;
    } else if (cmp_bits_dtype(bits, dtypes::LONG)) {
      return dtypes::LONG;
    } else if (cmp_bits_dtype(bits, dtypes::LONGLONG)) {
      return dtypes::LONGLONG;
    } else if (cmp_bits_dtype(bits, dtypes::BOOL)) {
      return dtypes::BOOL;
    }
  } else if (code == kDLUInt) {
    if (cmp_bits_dtype(bits, dtypes::UINT)) {
      return dtypes::UINT;
    } else if (cmp_bits_dtype(bits, dtypes::UBYTE)) {
      return dtypes::UBYTE;
    } else if (cmp_bits_dtype(bits, dtypes::USHORT)) {
      return dtypes::USHORT;
    } else if (cmp_bits_dtype(bits, dtypes::ULONG)) {
      return dtypes::ULONG;
    } else if (cmp_bits_dtype(bits, dtypes::ULONGLONG)) {
      return dtypes::ULONGLONG;
    }
  }

  NBLA_ERROR(error_code::type, "No matching types between NNabla dtypes and "
                               "DLPack DLDataType. code: %d, bits: %d",
             code, bits);
}

NdArrayPtr from_dlpack(DLManagedTensor *from) {
  auto to = make_shared<NdArray>();
  from_dlpack(from, to.get());
  return to;
}

void from_dlpack(DLManagedTensor *from, NdArray *to) {
  const auto shape = get_shape_with_contiguous_memory(from);
  to->reshape(shape, true);
  dynamic_cast<DlpackArray *>(
      to->cast(convert_dlpack_type_to_dtype(from->dl_tensor.dtype),
               DlpackArrayRegistry::create_context(from->dl_tensor), true))
      ->borrow(from);
}

class manager_ctx {
  shared_ptr<Array> array_;

public:
  manager_ctx(const shared_ptr<Array> &array) : array_(array) {}
};

void deleter(struct DLManagedTensor *self) {
  // Delete the members
  delete[] self->dl_tensor.shape;
  delete[] self->dl_tensor.strides;
  delete static_cast<manager_ctx *>(self->manager_ctx);

  // Finally delete itself.
  delete self;
}

DLManagedTensor *to_dlpack_impl(const shared_ptr<Array> &arr_ptr,
                                const Shape_t &shape, const Shape_t &strides) {
  const auto dtype = arr_ptr->dtype();
  const auto ctx = arr_ptr->context();

  // Create a DLTensor
  DLTensor dl_tensor;

  // data
  dl_tensor.data = arr_ptr->pointer<void *>();

  // DLContext
  // DLDeviceType
  dl_tensor.ctx.device_type =
      DlpackArrayRegistry::array_to_device_type(ctx.array_class);

  // Map string device_id in NNabla to int device_id in DLPack
  dl_tensor.ctx.device_id = 0;
  if (ctx.device_id != "") { // map "" to 0
    try {
      dl_tensor.ctx.device_id = std::stoi(ctx.device_id);
    } catch (...) {
      NBLA_ERROR(error_code::value, "device_id %s cannot be converted to "
                                    "integer.",
                 ctx.device_id.c_str());
    }
  }

  // ndim
  // It is a down-sized cast, but int type should be large enough for ndim
  dl_tensor.ndim = static_cast<int>(shape.size());

  // DLDataType
  // code
  dl_tensor.dtype.code = convert_dtype_to_dlpack_code(dtype);

  // bits
  dl_tensor.dtype.bits = sizeof_dtype(dtype) * 8;

  // lanes
  dl_tensor.dtype.lanes = 1;

  // shape
  dl_tensor.shape = new int64_t[dl_tensor.ndim]; // will deleted in deleter.
  for (int i = 0; i < dl_tensor.ndim; i++) {
    dl_tensor.shape[i] = shape[i];
  }

  // strides
  dl_tensor.strides = new int64_t[dl_tensor.ndim]; // will deleted in deleter.
  for (int i = 0; i < dl_tensor.ndim; i++) {
    dl_tensor.strides[i] = strides[i];
  }

  // byte_offset
  dl_tensor.byte_offset = 0;

  // Create a DLManagedTensor
  auto dl_managed_tensor = new DLManagedTensor();

  // dl_tensor
  dl_managed_tensor->dl_tensor = dl_tensor;

  // manager_ctx
  dl_managed_tensor->manager_ctx =
      static_cast<void *>(new manager_ctx(arr_ptr));

  // deleter
  dl_managed_tensor->deleter = deleter;

  return dl_managed_tensor;
}

DLManagedTensor *to_dlpack(NdArray *array) {
  const auto arr_ptr = array->array()->head_array_sp();
  return to_dlpack_impl(arr_ptr, array->shape(), array->strides());
}

DLManagedTensor *to_dlpack(NdArray *array, const dtypes dtype,
                           const Context &ctx) {
  const auto arr_ptr = array->cast_sp(dtype, ctx);
  return to_dlpack_impl(arr_ptr, array->shape(), array->strides());
}

void call_deleter(DLManagedTensor *dlp) { dlp->deleter(dlp); }
}
