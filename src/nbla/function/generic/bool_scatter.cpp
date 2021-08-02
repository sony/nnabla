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
#include <nbla/function/bool_scatter.hpp>
#include <nbla/function/utils/bool_indexing.hpp>
#include <nbla/variable.hpp>

#include <numeric>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BoolScatter);

template <typename T>
void BoolScatter<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  // sdata: NNZ, D_1, ..., D_M
  // mask: B_1, ..., B_N
  // gdata: B_1, ..., B_N, D_1, ..., D_M

  auto sdata = inputs[0];
  auto mask = inputs[1];
  auto gdata_inp = inputs.size() == 3 ? inputs[2] : nullptr;
  auto gdata_out = outputs[0];

  // Output shape
  auto sshape = sdata->shape();
  auto mshape = mask->shape();
  Shape_t gshape_out;
  for (auto &s : mshape)
    gshape_out.push_back(s);
  for (unsigned int i = 1; i < sshape.size(); i++)
    gshape_out.push_back(sshape[i]);
  gdata_out->reshape(gshape_out, true);

  // Inplace
  if (gdata_inp != nullptr) {
    NBLA_CHECK(gdata_inp->ndim() == gdata_out->ndim(), error_code::value,
               "Number of dimension of inplace output (%d) must be that of "
               "output (%d).",
               gdata_out->ndim(), gdata_inp->ndim());
    auto gshape_out = gdata_out->shape();
    auto gshape_inp = gdata_inp->shape();
    for (int i = 0; i < gdata_out->ndim(); ++i) {
      NBLA_CHECK(gshape_out[i] == gshape_inp[i], error_code::value,
                 "Shape of the inplaced output must be same as a computed "
                 "shape from inputs.");
    }
    gdata_out->data()->set_array(gdata_inp->data()->array());
  }
}

template <typename T>
void BoolScatter<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  auto mshape = inputs[1]->shape();
  auto gshape = outputs[0]->shape();
  auto B = inputs[1]->size();
  auto nnz = inputs[0]->shape()[0];
  auto D = inputs[0]->size() / nnz;

  auto inplace = (inputs.size() > 2);

  auto sdata = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto mask = inputs[1]->get_data_pointer<T>(this->ctx_);
  auto gdata = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, !inplace);

  auto kernel = inplace ? kernel_bool_scatter<T, false, true>
                        : kernel_bool_scatter<T, false, false>;
  kernel(D, B, nnz, gdata, sdata, mask);
}

template <typename T, bool accum = false>
inline void kernel_masked_identity(int B, int D, T *g_gdata_inp,
                                   const T *g_gdata_out, const T *mask) {
  for (int b = 0; b < B; ++b) {
    auto umask_b = T(mask[b] == T(0));
    for (int d = 0; d < D; ++d) {
      if (accum)
        g_gdata_inp[b * D + d] += umask_b * g_gdata_out[b * D + d];
      else
        g_gdata_inp[b * D + d] = umask_b * g_gdata_out[b * D + d];
    }
  }
}

template <typename T>
void BoolScatter<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }

  auto mshape = inputs[1]->shape();
  auto gshape = outputs[0]->shape();
  auto B = inputs[1]->size();
  auto nnz = inputs[0]->shape()[0];
  auto D = inputs[0]->size() / nnz;

  auto g_gdata = outputs[0]->get_grad_pointer<T>(this->ctx_);
  auto mask = inputs[1]->get_data_pointer<T>(this->ctx_);

  if (propagate_down[0]) {
    auto g_sdata =
        inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
    auto kernel =
        accum[0] ? kernel_bool_gather<T, true> : kernel_bool_gather<T, false>;
    kernel(D, B, nnz, g_sdata, g_gdata, mask);
  }

  // inplace
  if (inputs.size() > 2 && propagate_down[2]) {
    auto g_gdata_inp =
        inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[2]);
    auto kernel = accum[2] ? kernel_masked_identity<T, true>
                           : kernel_masked_identity<T, false>;
    kernel(B, D, g_gdata_inp, g_gdata, mask);
  }
}
}
