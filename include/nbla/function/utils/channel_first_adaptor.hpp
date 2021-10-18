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
using std::vector;

/**
 * @brief This class can be used to transform a variable memory format to
 * channel-first.
 *
 * Typical use case is a transformation from channel-last to channel-first
 * memory format in convolutional neural network.
 *
 * For example, let an input variable shape (32, 128, 128, 3), batch_axis ==
 * [0] and channel_axis == 3 (channel-last). Then this adaptor converts the
 * shape and memory format to (32, 3, 128, 128) by applying transpose function.
 *
 * Possible use case is a force use of channel-first implementation (layer
 * functions) in a channel-last network by sandwiching channel-first
 * implementation with this adaptor. The conceptual work flow (forward prop) is
 * following.
 * chennal-last layer -> forward_pre -> channel-first layer -> forward_post ->
 * channel-last layer
 * (Variebles are omitted and the same for backward prop)
 */
class ChannelFirstAdaptor {
  FunctionPtr pre_transpose_, post_transpose_;

public:
  /**
   * @brief Check wheather this adaptor is needed for the input.
   *
   * Returns `false` when the input is already channel-first format, otherwise
   * `true`.
   * The definition of channel-first memory format is when all of the following
   * conditions are met.
   * * `batch_axis` must be contiguous to the beginning of the shape.
   * * `channel_axis` must be right after the last axis of `batch_axis`.
   *
   * Examples of channel-first
   * * ndim: 4, batch_axis: [0], channel_axis: 1
   * * ndim: 2, batch_axis: [0], channel_axis: 1
   * * ndim: 2, batch_axis: [], channel_axis: 0
   * * ndim: 1, batch_axis: [], channel_axis: 0
   * * ndim: 6, batch_axis: [0, 1, 2], channel_axis: 3
   * * ndim: 6, batch_axis: [1, 0, 2], channel_axis: 3
   *
   * Examples of non channel-first
   * * ndim: 4, batch_axis: [0], channel_axis: 3
   * * ndim: 4, batch_axis: [0], channel_axis: 2
   * * ndim: 4, batch_axis: [1], channel_axis: 0
   * * ndim: 2, batch_axis: [], channel_axis: 1
   * * ndim: 6, batch_axis: [0, 1, 3], channel_axis: 2
   *
   * @param shape Shape of original input.
   * @param batch_axis List of integer corresponding to batch or outer axis.
   * Each axis must be unique and in range of [0, ndim).
   * @param channel_axis An integer corresponding to channel axis. This axis
   * must be in range of [0, ndim).
   * @return true
   * @return false
   */
  NBLA_API static bool need_adaptor(const Shape_t &shape,
                                    const vector<int> &batch_axis,
                                    const int channel_axis);

  /**
   * @brief Setup the adaptor.
   *
   * This method must be called before the use of `forward_(pre|post)` or
   * `backward_(pre|post)` methods. Setting up of transpose functions
   * (`(pre|post)_transpose_`) are performed internally.
   *
   * @param input_pre Variable pointer for `pre_transpose_` input.
   * @param output_pre Variable pointer for `pre_transpose_` output.
   * @param input_post Variable pointer for `post_transpose_` input.
   * @param output_post Variable pointer for `post_transpose_` output.
   * @param shape Shape of original input. This should be the same as the shape
   * of `input_pre`.
   * @param batch_axis List of integer corresponding to batch or outer axis.
   * Each axis must be in range of [0, ndim).
   * @param channel_axis An integer corresponding to channel axis. This axis
   * must be in range of [0, ndim).
   * @param ctx A compute backend descriptor.
   */
  NBLA_API void setup(Variable *input_pre, Variable *output_pre,
                      Variable *input_post, Variable *output_post,
                      const Shape_t &shape, const vector<int> &batch_axis,
                      const int channel_axis, const Context &ctx);

  /**
   * @brief Transform variable memory format to channel-first.
   *
   * @param input `pre_tranpose_` input.
   * @param output `pre_tranpose_` output.
   */
  NBLA_API void convert_to_channel_first(Variable *input, Variable *output);

  /**
   * @brief Transform variable memory format from channel-first to original one.
   *
   * @param input `post_tranpose_` input.
   * @param output `post_tranpose_` output.
   */
  NBLA_API void convert_from_channel_first(Variable *input, Variable *output);

  /**
   * @brief Backward execution for `forward_pre`.
   *
   * @param input `pre_tranpose_` input.
   * @param output `pre_tranpose_` output.
   * @param propagate_down Flag whether or not to perform backward propagation.
   * @param accum Flag whether or not to accumulate grad.
   */
  NBLA_API void convert_to_channel_first_backward(Variable *input,
                                                  Variable *output,
                                                  const bool propagate_down,
                                                  const bool accum);

  /**
   * @brief Backward execution for `forward_post`.
   *
   * @param input `post_tranpose_` input.
   * @param output `post_tranpose_` output.
   * @param propagate_down Flag whether or not to perform backward propagation.
   * @param accum Flag whether or not to accumulate grad.
   */
  NBLA_API void convert_from_channel_first_backward(Variable *input,
                                                    Variable *output,
                                                    const bool propagate_down,
                                                    const bool accum);
};

using ChannelFirstAdaptorPtr = std::shared_ptr<ChannelFirstAdaptor>;
}

#endif
