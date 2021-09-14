// Copyright 2021 Sony Corporation.
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

#ifndef __NBLA_FUNCTION_UTILS_CHANNEL_FIRST_ADAPTOR_HPP__
#define __NBLA_FUNCTION_UTILS_CHANNEL_FIRST_ADAPTOR_HPP__

#include <nbla/context.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/transpose.hpp>

namespace nbla {
class ChannelFirstAdaptor {
  FunctionPtr pre_transpose_, post_transpose_;

public:
  static bool need_adaptor(const Shape_t &shape,
                           const std::vector<int> &batch_axis,
                           const int channel_axis) {
    // Check batch_axis
    auto sorted_batch_axis = batch_axis;
    std::sort(sorted_batch_axis.begin(), sorted_batch_axis.end());
    for (int i = 0; i < sorted_batch_axis.size(); i++) {
      if (sorted_batch_axis[i] != i) {
        return true;
      }
    }

    // Check channel_axis
    if (channel_axis != sorted_batch_axis.size()) {
      return true;
    }

    // The input is channel-first memory format.
    return false;
  }

  void setup(Variable *input_pre, Variable *output_pre, Variable *input_post,
             Variable *output_post, const Shape_t &shape,
             const std::vector<int> &batch_axis, const int channel_axis,
             const Context &ctx) {
    const int ndim = shape.size();

    // Pre-transpose
    std::vector<int> pre_transpose_shape;
    for (const auto b : batch_axis) {
      pre_transpose_shape.push_back(b);
    }
    pre_transpose_shape.push_back(channel_axis);

    for (int i = 0; i < ndim; i++) {
      if (i == channel_axis) {
        continue;
      }

      // Check if `batch_axis` contains `i`.
      bool next = false;
      for (const auto b : batch_axis) {
        if (i == b) {
          next = true;
          break;
        }
      }
      if (next) {
        continue;
      }

      pre_transpose_shape.push_back(i);
    }

    pre_transpose_ = create_Transpose(ctx, pre_transpose_shape);
    pre_transpose_->setup({input_pre}, {output_pre});

    // Post-transpose
    std::vector<int> post_transpose_shape(ndim, -1);

    for (int i = 0; i < batch_axis.size(); i++) {
      post_transpose_shape[batch_axis[i]] = i;
    }

    post_transpose_shape[channel_axis] = batch_axis.size();

    int spacial_axis = batch_axis.size() + 1;
    for (int i = 0; i < ndim; i++) {
      if (post_transpose_shape[i] == -1) {
        post_transpose_shape[i] = spacial_axis;
        spacial_axis++;
      }
    }

    input_post->reshape(output_pre->shape(), true);
    post_transpose_ = create_Transpose(ctx, post_transpose_shape);
    post_transpose_->setup({input_post}, {output_post});
  }
  void forward_pre(Variable *input, Variable *output) {
    pre_transpose_->forward({input}, {output});
  }
  void forward_post(Variable *input, Variable *output) {
    post_transpose_->forward({input}, {output});
  }

  void backward_pre(Variable *input, Variable *output,
                    const bool propagate_down, const bool accum) {
    pre_transpose_->backward({input}, {output}, {propagate_down}, {accum});
  }
  void backward_post(Variable *input, Variable *output,
                     const bool propagate_down, const bool accum) {
    post_transpose_->backward({input}, {output}, {propagate_down}, {accum});
  }
};
}

#endif
