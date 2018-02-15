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
// 
// *WARNING*
// THIS FILE IS AUTO-GENERATED DUMMY CODE BY CODE GENERATOR.
// PLEASE IMPLEMENT REAL CODE AND DELETE THIS MESSAGE SOON.
// If you want to change dummy code, edit following files.
// - build-tools/code_generator/function_generator/generate_src_nbla_function_cpp.py
// - build-tools/code_generator/templates/src_nbla_function_cpp_template.cpp

/** GlobalAveragePooling
 */
#include <nbla/array.hpp>
#include <nbla/variable.hpp>
#include <nbla/function/global_average_pooling.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(GlobalAveragePooling);

template <typename T>
void GlobalAveragePooling<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
}

template <typename T>
void GlobalAveragePooling<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);

  const Shape_t inshape = inputs[0]->shape();
  const Shape_t outshape = outputs[0]->shape();
  const int CHANNEL_DIM = 2;
  NBLA_CHECK(inshape.size() > CHANNEL_DIM, error_code::value,
    "GlobalAveragePooling averages across the channel, "
    "so the input's shape must have a dimension larger than %d", CHANNEL_DIM);
  NBLA_CHECK(outshape.size() > CHANNEL_DIM, error_code::value,
    "GlobalAveragePooling averages across the channel, "
    "so the output's shape must have a dimension larger than %d", CHANNEL_DIM);
  NBLA_CHECK(inshape[CHANNEL_DIM] == outshape[CHANNEL_DIM], error_code::value,
    "Input channel and output channel size must match");
  const int ndim = outshape[0];
  const int chandim = outshape[1];
  const int in_w = inshape[2];
  const int in_h = inshape[3];
  const int in_wh = in_w*in_h;
  const int in_n_ofs = in_wh*chandim;

  for (int n = 0; n < ndim; ++n) {
    const T *xchan = &x[n * in_n_ofs];
    T *ychan = &y[n * chandim];
    for (int c = 0; c < chandim; ++c) {
      const T *ximg = &xchan[c*in_wh];
      // calculate average of each channel
      T avg = std::accumulate(ximg, ximg+in_wh, T(0)) / (T)in_wh;
      ychan[c] = avg;
    }
  }
}

template <typename T>
void GlobalAveragePooling<T>::backward_impl(const Variables &inputs, const Variables &outputs,
					     const vector<bool> &propagate_down,
					     const vector<bool> &accum) {
  if (!propagate_down[0])
    return;

  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const Shape_t inshape = inputs[0]->shape();
  const Shape_t outshape = outputs[0]->shape();
  const int ndim = outshape[0];
  const int chandim = outshape[1];
  const int in_w = inshape[2];
  const int in_h = inshape[3];
  const int in_wh = in_w*in_h;
  const int in_n_ofs = in_wh*chandim;
  const bool accumulate = accum[0];

  if (accumulate) {
    for (int n = 0; n < ndim; ++n) {
      T *dxchan = &dx[n * in_n_ofs];
      const T *dychan = &dy[n * chandim];
      for (int c = 0; c < chandim; ++c) {
        T *dximg = &dxchan[c*in_wh];
        const T dyval = dychan[c] / (T)in_wh;
        std::transform(dximg, dximg+in_wh, dximg, [=](T val){return val+dyval;});
      }
    }
  } else {
    for (int n = 0; n < ndim; ++n) {
      T *dxchan = &dx[n * in_n_ofs];
      const T *dychan = &dy[n * chandim];
      for (int c = 0; c < chandim; ++c) {
        T *dximg = &dxchan[c*in_wh];
        const T dyval = dychan[c] / (T)in_wh;
        std::fill(dximg, dximg+in_wh, dyval);
      }
    }
  }
}

// Template instantiation
template class GlobalAveragePooling<float>;
} // namespace nbla
