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
#include <nbla/function/mod2.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Mod2, bool);

namespace {
class FloatModBinaryOp : public BaseBinaryOp {
public:
  template <typename T> inline T operator()(const T x0, const T x1) {
    return std::fmod(x0, x1);
  }
};

class IntegerModBinaryOp : public BaseBinaryOp {
public:
  template <typename T> inline T operator()(const T x0, const T x1) {
    T ret = x0 % x1;
    if ((ret > 0 && x1 < 0) || (ret < 0 && x1 > 0)) {
      ret += x1;
    }
    return ret;
  }
};

template <typename T, typename BinaryOp>
void mod(const Size_t size, const T *x0, const T *x1, T *y, BinaryOp op,
         const Size_t ndim, const Size_t *strides_x0, const Size_t *strides_x1,
         const Size_t *strides_y, const Size_t *shape_y) {
  for (Size_t idx = 0; idx < size; ++idx) {
    Size_t idx0 = 0;
    Size_t idx1 = 0;
    for (Size_t i = 0; i < ndim; ++i) {
      Size_t dim_idx = (idx / strides_y[i]) % shape_y[i];
      idx0 += dim_idx * strides_x0[i];
      idx1 += dim_idx * strides_x1[i];
    }
    y[idx] = op(x0[idx0], x1[idx1]);
  }
}

template <typename T>
void floating_point_mod(const Variables &inputs, const Variables &outputs,
                        const Context ctx, const Size_t ndim,
                        const Size_t *strides_x0, const Size_t *strides_x1,
                        const Size_t *strides_y, const Size_t *shape_y) {
  const T *x0 = inputs[0]->get_data_pointer<T>(ctx);
  const T *x1 = inputs[1]->get_data_pointer<T>(ctx);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(ctx, true);

  FloatModBinaryOp op;
  mod<T, FloatModBinaryOp>(outputs[0]->size(), x0, x1, y, op, ndim, strides_x0,
                           strides_x1, strides_y, shape_y);
}

template <typename T>
void integer_mod(const Variables &inputs, const Variables &outputs,
                 const Context ctx, const Size_t ndim, const Size_t *strides_x0,
                 const Size_t *strides_x1, const Size_t *strides_y,
                 const Size_t *shape_y, const bool fmod) {
  const T *x0 = inputs[0]->get_data_pointer<T>(ctx);
  const T *x1 = inputs[1]->get_data_pointer<T>(ctx);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(ctx, true);

  if (fmod) {
    FloatModBinaryOp op;
    mod<T, FloatModBinaryOp>(outputs[0]->size(), x0, x1, y, op, ndim,
                             strides_x0, strides_x1, strides_y, shape_y);
  } else {
    IntegerModBinaryOp op;
    mod<T, IntegerModBinaryOp>(outputs[0]->size(), x0, x1, y, op, ndim,
                               strides_x0, strides_x1, strides_y, shape_y);
  }
}
}

template <typename T>
void Mod2<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const Size_t *strides_x0 =
      this->strides_x0_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *strides_x1 =
      this->strides_x1_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *strides_y =
      this->strides_y_.template get_data_pointer<Size_t>(this->ctx_);
  const Size_t *shape_y =
      this->shape_y_.template get_data_pointer<Size_t>(this->ctx_);

  const dtypes dtype = inputs[0]->data()->array()->dtype();
  switch (dtype) {
  case dtypes::HALF:
    floating_point_mod<Half>(inputs, outputs, this->ctx_,
                             this->compressed_ndim_, strides_x0, strides_x1,
                             strides_y, shape_y);
    break;
  case dtypes::FLOAT:
    floating_point_mod<float>(inputs, outputs, this->ctx_,
                              this->compressed_ndim_, strides_x0, strides_x1,
                              strides_y, shape_y);
    break;
  case dtypes::DOUBLE:
    floating_point_mod<double>(inputs, outputs, this->ctx_,
                               this->compressed_ndim_, strides_x0, strides_x1,
                               strides_y, shape_y);
    break;
  case dtypes::UBYTE:
    integer_mod<unsigned char>(inputs, outputs, this->ctx_,
                               this->compressed_ndim_, strides_x0, strides_x1,
                               strides_y, shape_y, fmod_);
    break;
  case dtypes::USHORT:
    integer_mod<unsigned short>(inputs, outputs, this->ctx_,
                                this->compressed_ndim_, strides_x0, strides_x1,
                                strides_y, shape_y, fmod_);
    break;
  case dtypes::UINT:
    integer_mod<unsigned int>(inputs, outputs, this->ctx_,
                              this->compressed_ndim_, strides_x0, strides_x1,
                              strides_y, shape_y, fmod_);
    break;
  case dtypes::ULONG:
    integer_mod<unsigned long>(inputs, outputs, this->ctx_,
                               this->compressed_ndim_, strides_x0, strides_x1,
                               strides_y, shape_y, fmod_);
    break;
  case dtypes::ULONGLONG:
    integer_mod<unsigned long long>(inputs, outputs, this->ctx_,
                                    this->compressed_ndim_, strides_x0,
                                    strides_x1, strides_y, shape_y, fmod_);
    break;
  case dtypes::BYTE:
    integer_mod<char>(inputs, outputs, this->ctx_, this->compressed_ndim_,
                      strides_x0, strides_x1, strides_y, shape_y, fmod_);
    break;
  case dtypes::SHORT:
    integer_mod<short>(inputs, outputs, this->ctx_, this->compressed_ndim_,
                       strides_x0, strides_x1, strides_y, shape_y, fmod_);
    break;
  case dtypes::INT:
    integer_mod<int>(inputs, outputs, this->ctx_, this->compressed_ndim_,
                     strides_x0, strides_x1, strides_y, shape_y, fmod_);
    break;
  case dtypes::LONG:
    integer_mod<long>(inputs, outputs, this->ctx_, this->compressed_ndim_,
                      strides_x0, strides_x1, strides_y, shape_y, fmod_);
    break;
  case dtypes::LONGLONG:
    integer_mod<long long>(inputs, outputs, this->ctx_, this->compressed_ndim_,
                           strides_x0, strides_x1, strides_y, shape_y, fmod_);
    break;
  default:
    NBLA_ERROR(error_code::type, "inputs[0] has unsupported dtype: %s.",
               dtype_to_string(dtype).c_str());
    break;
  }
}

template <typename T>
void Mod2<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  NBLA_ERROR(error_code::not_implemented,
             "Mod2<T>::backward is currently not implemented.");
}
}
