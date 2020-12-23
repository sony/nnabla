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

#include <numeric>

#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/pack_padded_sequence.hpp>
#include <nbla/function/utils/rnn.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(PackPaddedSequence, bool);

template <typename U>
void PackPaddedSequence<U>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
  // inputs[0]:  padded_sequence (T, B, D_1, ..., D_M)
  // inputs[1]:  lengths         (B)
  // outputs[0]: packed_sequence (N, D_1, ..., D_M)
  // outputs[1]: batch_sizes     (T)
  // N = sum(lengths) = sum(batch_sizes)

  nbla::Context cpu_ctx{{"cpu:int"}, "CpuCachedArray", "0"};
  auto padded_sequence = inputs[0];
  auto ndim = padded_sequence->ndim();
  auto ishape = padded_sequence->shape();
  auto lengths = inputs[1];

  NBLA_CHECK(ndim >= 2, error_code::value,
             "Ndim of inputs[0] (%d) must be greater than or equal to 2.",
             ndim);
  NBLA_CHECK(lengths->ndim() == 1, error_code::value,
             "Ndim of inputs[1] (%d) must be 1.", lengths->ndim());

  auto data_lengths = lengths->get_data_pointer<int>(cpu_ctx);
  vector<int> tmp_data_lengths(data_lengths, data_lengths + lengths->size());
  int N = std::accumulate(tmp_data_lengths.begin(), tmp_data_lengths.end(), 0,
                          std::plus<int>());
  Shape_t oshape{N};
  if (ndim > 2)
    oshape.insert(oshape.end(), ishape.begin() + 2, ishape.end());
  auto T =
      batch_first_ ? padded_sequence->shape()[1] : padded_sequence->shape()[0];
  outputs[0]->reshape(oshape, true);
  outputs[1]->reshape(Shape_t{T}, true);
}

template <typename U>
void PackPaddedSequence<U>::forward_impl(const Variables &inputs,
                                         const Variables &outputs) {
  nbla::Context cpu_ctx{{"cpu:int"}, "CpuCachedArray", "0"};
  auto padded_sequence = inputs[0];
  auto lengths = inputs[1];
  auto packed_sequence = outputs[0];
  auto batch_sizes = outputs[1];

  //   N = packed_sequence->shape()[0];
  auto T = batch_sizes->shape()[0];
  auto B = lengths->shape()[0];
  auto D = packed_sequence->ndim() == 1 ? 1 : packed_sequence->size(1);
  auto ishape0 = padded_sequence->shape();
  auto oshape0 = packed_sequence->shape();
  auto data_padded_sequence = padded_sequence->get_data_pointer<U>(ctx_);
  auto data_lengths = lengths->get_data_pointer<int>(cpu_ctx);
  auto data_packed_sequence =
      packed_sequence->cast_data_and_get_pointer<U>(ctx_);
  auto data_batch_sizes = batch_sizes->cast_data_and_get_pointer<int>(cpu_ctx);
  namespace rnn = function::utils::rnn;
  // Compute batch_sizes
  rnn::compute_batch_sizes(data_lengths, lengths->size(), data_batch_sizes);
  // Pack
  if (batch_first_) {
    rnn::pack_batch_first<U, false>(data_padded_sequence, data_batch_sizes,
                                    data_packed_sequence, T, B, D);
  } else {
    rnn::pack<U, false>(data_padded_sequence, data_batch_sizes,
                        data_packed_sequence, T, B, D);
  }
}

template <typename U>
void PackPaddedSequence<U>::backward_impl(const Variables &inputs,
                                          const Variables &outputs,
                                          const vector<bool> &propagate_down,
                                          const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  nbla::Context cpu_ctx{{"cpu:int"}, "CpuCachedArray", "0"};
  auto padded_sequence = inputs[0];
  auto lengths = inputs[1];
  auto packed_sequence = outputs[0];
  auto batch_sizes = outputs[1];

  // TODO: batch_first
  //   N = packed_sequence->shape()[0];
  auto T = batch_sizes->shape()[0];
  auto B = lengths->shape()[0];
  auto D = packed_sequence->ndim() == 1 ? 1 : packed_sequence->size(1);
  auto ishape0 = padded_sequence->shape();
  auto oshape0 = packed_sequence->shape();
  auto grad_padded_sequence =
      padded_sequence->cast_grad_and_get_pointer<U>(ctx_, !accum[0]);
  auto grad_packed_sequence = packed_sequence->get_grad_pointer<U>(ctx_);
  auto data_batch_sizes = batch_sizes->get_data_pointer<int>(cpu_ctx);
  // Unpack
  namespace rnn = function::utils::rnn;
  if (accum[0]) {
    if (batch_first_) {
      rnn::unpack_batch_first<U, true>(grad_packed_sequence, data_batch_sizes,
                                       grad_padded_sequence, T, B, D);
    } else {
      rnn::unpack<U, true>(grad_packed_sequence, data_batch_sizes,
                           grad_padded_sequence, T, B, D);
    }
  } else {
    if (batch_first_) {
      rnn::unpack_batch_first<U, false>(grad_packed_sequence, data_batch_sizes,
                                        grad_padded_sequence, T, B, D);
    } else {
      rnn::unpack<U, false>(grad_packed_sequence, data_batch_sizes,
                            grad_padded_sequence, T, B, D);
    }
  }
}
}
