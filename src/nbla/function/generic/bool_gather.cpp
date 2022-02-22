// Copyright 2021 Sony Group Corporation.
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
#include <nbla/function/bool_gather.hpp>
#include <nbla/function/sum.hpp>
#include <nbla/function/utils/bool_indexing.hpp>
#include <nbla/imperative.hpp>
#include <nbla/variable.hpp>

#include <numeric>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BoolGather);

template <typename T>
void BoolGather<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  // gdata: B_1, ..., B_N, D_1, ..., D_M
  // mask: B_1, ..., B_N
  // sdata: NNZ, D_1, ..., D_M

  // Setup_impl must be called all time
  auto gdata = inputs[0];
  auto mask = inputs[1];
  auto sdata = outputs[0];
  auto gshape = gdata->shape();
  auto mshape = mask->shape();

  // Error check
  NBLA_CHECK(gdata->ndim() >= mask->ndim(), error_code::value,
             "input.ndim (%d) >= mask.ndim (%d)", gdata->ndim(), mask->ndim());
  for (int i = 0; i < mask->ndim(); ++i) {
    NBLA_CHECK(mshape[i] == gshape[i], error_code::value,
               "mask.shape must be equal to input.shape[:mask.ndim]. "
               "mask.shape[%d] (%d) != input.shape[%d] (%d)",
               i, mshape[i], i, gshape[i]);
  }

  // TODO: add F.not_equal_scalar == 0
  // Number of non-zero elements
  vector<int> axes(mask->ndim());
  std::iota(axes.begin(), axes.end(), 0);
  f_sum_ = create_Sum(this->ctx_, axes, false);
  Variable nnz({1});
  nbla::execute(f_sum_, {mask}, {&nnz});
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  auto data_nnz = nnz.get_data_pointer<float>(cpu_ctx);

  // Output shape
  auto B = int(*data_nnz);
  if (gdata->ndim() != mask->ndim()) {
    auto D = std::accumulate(gshape.begin() + mask->ndim(), gshape.end(), 1,
                             std::multiplies<int>());
    Shape_t sshape({B, D});
    sdata->reshape(sshape, true);
  } else {
    Shape_t sshape({B});
    sdata->reshape(sshape, true);
  }
}

template <typename T>
void BoolGather<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  // Outputs
  auto mshape = inputs[1]->shape();
  auto B =
      std::accumulate(mshape.begin(), mshape.end(), 1, std::multiplies<int>());
  auto nnz = outputs[0]->shape()[0];
  auto D = outputs[0]->size() / nnz;
  T *sdata = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  const T *gdata = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *mask = inputs[1]->get_data_pointer<T>(this->ctx_);

  kernel_bool_gather<T>(D, B, nnz, sdata, gdata, mask);
}

template <typename T>
void BoolGather<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  auto mshape = inputs[1]->shape();
  auto B =
      std::accumulate(mshape.begin(), mshape.end(), 1, std::multiplies<int>());
  auto nnz = outputs[0]->shape()[0];
  auto D = outputs[0]->size() / nnz;
  const T *g_sdata = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *mask = inputs[1]->get_data_pointer<T>(this->ctx_);

  if (propagate_down[0]) {
    auto g_gdata =
        inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
    auto kernel =
        accum[0] ? kernel_bool_scatter<T, true> : kernel_bool_scatter<T, false>;
    kernel(D, B, nnz, g_gdata, g_sdata, mask);
  }
}
}
