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

#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/global_average_pooling.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <numeric>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(GlobalAveragePooling);

template <typename T>
void GlobalAveragePooling<T>::setup_impl(const Variables &inputs,
                                         const Variables &outputs) {
  const Shape_t inshape = inputs[0]->shape();
  const int in_dim = inshape.size();
  const int MIN_DIM = 2;
  NBLA_CHECK(in_dim >= MIN_DIM, error_code::value,
             "GlobalAveragePooling averages across the channel, "
             "so the input's shape must have a dimension equal to or larger "
             "than %d. actual: %d",
             MIN_DIM, in_dim);
  Shape_t shape_out;
  shape_out.push_back(inshape[0]);
  shape_out.push_back(inshape[1]);
  shape_out.push_back(1);
  shape_out.push_back(1);
  outputs[0]->reshape(shape_out, true);
}

template <typename T>
void GlobalAveragePooling<T>::forward_impl(const Variables &inputs,
                                           const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);

  const Shape_t inshape = inputs[0]->shape();
  const Shape_t outshape = outputs[0]->shape();
  const int in_dim = inshape.size();
  const int ndim = outshape[0];
  const int chandim = outshape[1];
  const int in_h = in_dim >= 3 ? inshape[2] : 1;
  const int in_w = in_dim >= 4 ? inshape[3] : 1;
  const int in_wh = in_w * in_h;
  const int in_n_ofs = in_wh * chandim;

  for (int n = 0; n < ndim; ++n) {
    const T *xchan = &x[n * in_n_ofs];
    T *ychan = &y[n * chandim];
    for (int c = 0; c < chandim; ++c) {
      const T *ximg = &xchan[c * in_wh];
      // calculate average of each channel
      T avg = std::accumulate(ximg, ximg + in_wh, T(0)) / (T)in_wh;
      ychan[c] = avg;
    }
  }
}

template <typename T>
void GlobalAveragePooling<T>::backward_impl(const Variables &inputs,
                                            const Variables &outputs,
                                            const vector<bool> &propagate_down,
                                            const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const Shape_t inshape = inputs[0]->shape();
  const Shape_t outshape = outputs[0]->shape();
  const int in_dim = inshape.size();
  const int ndim = outshape[0];
  const int chandim = outshape[1];
  const int in_h = in_dim >= 3 ? inshape[2] : 1;
  const int in_w = in_dim >= 4 ? inshape[3] : 1;
  const int in_wh = in_w * in_h;
  const int in_n_ofs = in_wh * chandim;
  const bool accumulate = accum[0];

  if (accumulate) {
    for (int n = 0; n < ndim; ++n) {
      T *dxchan = &dx[n * in_n_ofs];
      const T *dychan = &dy[n * chandim];
      for (int c = 0; c < chandim; ++c) {
        T *dximg = &dxchan[c * in_wh];
        const T dyval = dychan[c] / (T)in_wh;
        std::transform(dximg, dximg + in_wh, dximg,
                       [=](T val) { return val + dyval; });
      }
    }
  } else {
    for (int n = 0; n < ndim; ++n) {
      T *dxchan = &dx[n * in_n_ofs];
      const T *dychan = &dy[n * chandim];
      for (int c = 0; c < chandim; ++c) {
        T *dximg = &dxchan[c * in_wh];
        const T dyval = dychan[c] / (T)in_wh;
        std::fill(dximg, dximg + in_wh, dyval);
      }
    }
  }
}
}
