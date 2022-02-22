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

#include <nbla/function/utils/channel_first_adaptor.hpp>

namespace nbla {

bool ChannelFirstAdaptor::need_adaptor(const Shape_t &shape,
                                       const vector<int> &batch_axis,
                                       const int channel_axis) {
  // Check batch_axis
  // batch_axis must be contiguous to the beginning of the shape.
  // eg. (N1, N2, N3, C, H, W)
  auto sorted_batch_axis = batch_axis;
  std::sort(sorted_batch_axis.begin(), sorted_batch_axis.end());
  for (size_t i = 0; i < sorted_batch_axis.size(); i++) {
    if (sorted_batch_axis[i] != i) {
      return true;
    }
  }

  // Check channel_axis
  if (channel_axis != (int)sorted_batch_axis.size()) {
    return true;
  }

  // The input is channel-first memory format.
  return false;
}

void ChannelFirstAdaptor::setup(Variable *input_pre, Variable *output_pre,
                                Variable *input_post, Variable *output_post,
                                const Shape_t &shape,
                                const vector<int> &batch_axis,
                                const int channel_axis, const Context &ctx) {
  const int ndim = shape.size();

  // --------------------------------
  // Setup pre-transpose
  // --------------------------------
  vector<int> pre_transpose_shape;
  // Batch axis
  for (const auto b : batch_axis) {
    pre_transpose_shape.push_back(b);
  }

  // Channel axis
  pre_transpose_shape.push_back(channel_axis);

  // Spacital axis
  for (int i = 0; i < ndim; i++) {
    // Ignore channel axis
    if (i == channel_axis) {
      continue;
    }
    // Ignore batch axis
    const auto result = std::find(batch_axis.begin(), batch_axis.end(), i);
    if (result != batch_axis.end()) { // Found `i`
      continue;
    }

    // Append spacial axis
    pre_transpose_shape.push_back(i);
  }

  // Setup transpose function
  pre_transpose_ = create_Transpose(ctx, pre_transpose_shape);
  pre_transpose_->setup({input_pre}, {output_pre});

  // --------------------------------
  // Setup post-transpose
  // --------------------------------
  vector<int> post_transpose_shape(ndim, -1);

  // Batch axis
  for (size_t i = 0; i < batch_axis.size(); i++) {
    post_transpose_shape[batch_axis[i]] = i;
  }

  // Channel axis
  post_transpose_shape[channel_axis] = batch_axis.size();

  // Spacial axis
  int spacial_axis = batch_axis.size() + 1;
  for (auto i = 0; i < ndim; i++) {
    if (post_transpose_shape[i] == -1) {
      post_transpose_shape[i] = spacial_axis;
      spacial_axis++;
    }
  }

  // Setup transpose function
  input_post->reshape(output_pre->shape(), true);
  post_transpose_ = create_Transpose(ctx, post_transpose_shape);
  post_transpose_->setup({input_post}, {output_post});
}

void ChannelFirstAdaptor::convert_to_channel_first(Variable *input,
                                                   Variable *output) {
  pre_transpose_->forward({input}, {output});
}

void ChannelFirstAdaptor::convert_from_channel_first(Variable *input,
                                                     Variable *output) {
  post_transpose_->forward({input}, {output});
}

void ChannelFirstAdaptor::convert_to_channel_first_backward(
    Variable *input, Variable *output, const bool propagate_down,
    const bool accum) {
  pre_transpose_->backward({input}, {output}, {propagate_down}, {accum});
}

void ChannelFirstAdaptor::convert_from_channel_first_backward(
    Variable *input, Variable *output, const bool propagate_down,
    const bool accum) {
  post_transpose_->backward({input}, {output}, {propagate_down}, {accum});
}
}
