// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/bit_shift.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BitShift, const string &);

class ShiftLeftBinaryOp : public BaseBinaryOp {
public:
  template <typename T> inline T operator()(const T x0, const T x1) {
    T ret = T(0);
    if (x1 >= 0 && x1 < sizeof(T) * 8) {
      ret = x0 << x1;
    }
    return ret;
  }
};

class ShiftRightBinaryOp : public BaseBinaryOp {
public:
  template <typename T> inline T operator()(const T x0, const T x1) {
    T ret = T(0);
    if (x1 >= 0 && x1 < sizeof(T) * 8) {
      ret = x0 >> x1;
    }
    return ret;
  }
};

template <typename T>
void dispatch_shift_direction(const Variables &inputs, const Variables &outputs,
                              const Context ctx, const Size_t ndim,
                              const Size_t *strides_x0,
                              const Size_t *strides_x1, const Size_t *strides_y,
                              const Size_t *shape_y, const bool shift_left) {
  const T *x0 = inputs[0]->get_data_pointer<T>(ctx);
  const T *x1 = inputs[1]->get_data_pointer<T>(ctx);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(ctx, true);

  if (shift_left) {
    ShiftLeftBinaryOp op;
    transform_binary(outputs[0]->size(), x0, x1, y, op, ndim, strides_x0,
                     strides_x1, strides_y, shape_y);
  } else {
    ShiftRightBinaryOp op;
    transform_binary(outputs[0]->size(), x0, x1, y, op, ndim, strides_x0,
                     strides_x1, strides_y, shape_y);
  }
}

template <typename T>
void BitShift<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  BaseTransformBinary::setup_impl(inputs, outputs);

  NBLA_CHECK(direction_ == "LEFT" || direction_ == "RIGHT", error_code::value,
             "Unsupported direction: %s.", direction_.c_str());

  if (direction_ == "LEFT") {
    shift_left_ = true;
  } else {
    shift_left_ = false;
  }
}

template <typename T>
void BitShift<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  const dtypes dtype = inputs[0]->data()->array()->dtype();

  const Size_t *strides_x0 =
      this->strides_x0_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *strides_x1 =
      this->strides_x1_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *strides_y =
      this->strides_y_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *shape_y =
      this->shape_y_.template get_data_pointer<Size_t>(this->ctx_);

  switch (dtype) {
  case dtypes::UBYTE:
    dispatch_shift_direction<unsigned char>(
        inputs, outputs, this->ctx_, this->compressed_ndim_, strides_x0,
        strides_x1, strides_y, shape_y, shift_left_);
    break;
  case dtypes::USHORT:
    dispatch_shift_direction<unsigned short>(
        inputs, outputs, this->ctx_, this->compressed_ndim_, strides_x0,
        strides_x1, strides_y, shape_y, shift_left_);
    break;
  case dtypes::UINT:
    dispatch_shift_direction<unsigned int>(
        inputs, outputs, this->ctx_, this->compressed_ndim_, strides_x0,
        strides_x1, strides_y, shape_y, shift_left_);
    break;
  case dtypes::ULONG:
    dispatch_shift_direction<unsigned long>(
        inputs, outputs, this->ctx_, this->compressed_ndim_, strides_x0,
        strides_x1, strides_y, shape_y, shift_left_);
    break;
  case dtypes::ULONGLONG:
    dispatch_shift_direction<unsigned long long>(
        inputs, outputs, this->ctx_, this->compressed_ndim_, strides_x0,
        strides_x1, strides_y, shape_y, shift_left_);
    break;
  default:
    NBLA_ERROR(error_code::type, "inputs[0] has unsupported dtype: %s.",
               dtype_to_string(dtype).c_str());
    break;
  }
}

template <typename T>
void BitShift<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  NBLA_ERROR(error_code::not_implemented,
             "BitShift<T>::backward is currently not implemented.");
}
}
