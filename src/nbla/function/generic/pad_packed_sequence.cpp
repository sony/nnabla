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
#include <nbla/function/pad_packed_sequence.hpp>
#include <nbla/function/utils/rnn.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(PadPackedSequence, bool, float, int);

template <typename U>
void PadPackedSequence<U>::setup_impl(const Variables &inputs,
                                      const Variables &outputs) {
  // inputs[0]:  packed_sequence (N, D_1, ..., D_M)
  // inputs[1]:  batch_sizes     (T)
  // outputs[0]: padded_sequence (T, B, D_1, ..., D_M)
  // outputs[1]: lengths         (B)
  // N = sum(lengths) = sum(batch_sizes)

  nbla::Context cpu_ctx{{"cpu:int"}, "CpuCachedArray", "0"};
  auto packed_sequence = inputs[0];
  auto ndim = packed_sequence->ndim();
  auto ishape = packed_sequence->shape();
  auto batch_sizes = inputs[1];
  auto T = batch_sizes->shape()[0];

  NBLA_CHECK(ndim >= 1, error_code::value,
             "Ndim of inputs[0] (%d) must be greater than or equal to 1.",
             ndim);
  NBLA_CHECK(batch_sizes->ndim() == 1, error_code::value,
             "Ndim of inputs[1] (%d) must be 1.", batch_sizes->ndim());

  auto data_batch_sizes = batch_sizes->get_data_pointer<int>(cpu_ctx);
  vector<int> tmp_data_batch_sizes(data_batch_sizes,
                                   data_batch_sizes + batch_sizes->size());
  auto d = std::distance(tmp_data_batch_sizes.begin(),
                         std::max_element(tmp_data_batch_sizes.begin(),
                                          tmp_data_batch_sizes.end()));
  auto B = data_batch_sizes[d];
  auto T0 = (total_length_ > T) ? total_length_ : T;
  auto D0 = batch_first_ ? B : T0;
  auto D1 = batch_first_ ? T0 : B;
  Shape_t oshape{D0, D1};
  if (ndim > 1)
    oshape.insert(oshape.end(), ishape.begin() + 1, ishape.end());
  outputs[0]->reshape(oshape, true);
  outputs[1]->reshape(Shape_t{B}, true);
}

template <typename U>
void PadPackedSequence<U>::forward_impl(const Variables &inputs,
                                        const Variables &outputs) {
  nbla::Context cpu_ctx{{"cpu:int"}, "CpuCachedArray", "0"};
  auto packed_sequence = inputs[0];
  auto batch_sizes = inputs[1];
  auto padded_sequence = outputs[0];
  auto lengths = outputs[1];

  //   N = packed_sequence->shape()[0];
  auto T = batch_sizes->shape()[0];
  auto B = lengths->shape()[0];
  auto D = packed_sequence->ndim() == 1 ? 1 : packed_sequence->size(1);
  auto TL = this->total_length_;
  auto ishape0 = packed_sequence->shape();
  auto oshape0 = padded_sequence->shape();
  auto data_packed_sequence = packed_sequence->get_data_pointer<U>(ctx_);
  auto data_batch_sizes = batch_sizes->get_data_pointer<int>(cpu_ctx);
  auto data_padded_sequence =
      padded_sequence->cast_data_and_get_pointer<U>(ctx_);
  auto data_lengths = lengths->cast_data_and_get_pointer<int>(cpu_ctx);

  namespace rnn = function::utils::rnn;
  // Compute lengths
  rnn::compute_lengths(data_batch_sizes, batch_sizes->size(), data_lengths);
  // Unpack
  if (batch_first_) {
    if (TL > T)
      rnn::unpack_batch_first(data_packed_sequence, data_batch_sizes,
                              data_padded_sequence, T, B, D, TL);
    else
      rnn::unpack_batch_first(data_packed_sequence, data_batch_sizes,
                              data_padded_sequence, T, B, D);
  } else {
    if (TL > T)
      rnn::unpack(data_packed_sequence, data_batch_sizes, data_padded_sequence,
                  T, B, D, TL);
    else
      rnn::unpack(data_packed_sequence, data_batch_sizes, data_padded_sequence,
                  T, B, D);
  }
}

template <typename U>
void PadPackedSequence<U>::backward_impl(const Variables &inputs,
                                         const Variables &outputs,
                                         const vector<bool> &propagate_down,
                                         const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  nbla::Context cpu_ctx{{"cpu:int"}, "CpuCachedArray", "0"};
  auto packed_sequence = inputs[0];
  auto batch_sizes = inputs[1];
  auto padded_sequence = outputs[0];
  auto lengths = outputs[1];

  //   N = packed_sequence->shape()[0];
  auto T = batch_sizes->shape()[0];
  auto B = lengths->shape()[0];
  auto D = packed_sequence->ndim() == 1 ? 1 : packed_sequence->size(1);
  auto TL = this->total_length_;
  auto ishape0 = packed_sequence->shape();
  auto oshape0 = padded_sequence->shape();
  auto grad_packed_sequence =
      packed_sequence->cast_grad_and_get_pointer<U>(ctx_, !accum[0]);
  auto data_batch_sizes = batch_sizes->get_data_pointer<int>(cpu_ctx);
  auto grad_padded_sequence = padded_sequence->get_grad_pointer<U>(ctx_);

  // Pack
  namespace rnn = function::utils::rnn;
  if (accum[0]) {
    if (batch_first_) {
      if (TL > T)
        rnn::pack_batch_first<U, true>(grad_padded_sequence, data_batch_sizes,
                                       grad_packed_sequence, T, B, D, TL);
      else
        rnn::pack_batch_first<U, true>(grad_padded_sequence, data_batch_sizes,
                                       grad_packed_sequence, T, B, D);
    } else {
      rnn::pack<U, true>(grad_padded_sequence, data_batch_sizes,
                         grad_packed_sequence, T, B, D);
    }
  } else {
    if (batch_first_) {
      if (TL > T)
        rnn::pack_batch_first<U, false>(grad_padded_sequence, data_batch_sizes,
                                        grad_packed_sequence, T, B, D, TL);
      else
        rnn::pack_batch_first<U, false>(grad_padded_sequence, data_batch_sizes,
                                        grad_packed_sequence, T, B, D);
    } else {
      rnn::pack<U, false>(grad_padded_sequence, data_batch_sizes,
                          grad_packed_sequence, T, B, D);
    }
  }
}
}
