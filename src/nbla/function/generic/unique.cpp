// Copyright 2023 Sony Group Corporation.
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
#include <nbla/auto_forward.hpp>
#include <nbla/common.hpp>
#include <nbla/function/transpose.hpp>
#include <nbla/function/unique.hpp>
#include <nbla/utils/axis_utils.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Unique, bool, int, bool, bool, bool, bool);

template <typename T>
static void
create_outputs(Context ctx, const Variables &inputs, const Variables &outputs,
               const vector<int> &indices_val, const vector<int> &inverse_val,
               const vector<int> &counts_val, bool flatten, int axis,
               bool sorted, bool with_index, bool with_inverse,
               bool with_counts, bool is_recompute) {
  const int num_unique = indices_val.size();

  // If sorted is False, do argsort of indices
  vector<int> output_order(num_unique);
  std::iota(output_order.begin(), output_order.end(), 0);
  if (!sorted) {
    std::sort(output_order.begin(), output_order.end(),
              [&](const int &a, const int &b) {
                return indices_val[a] < indices_val[b];
              });
  }

  // Create output y
  if (flatten) {
    const T *x = inputs[0]->get_data_pointer<T>(ctx);
    Shape_t y_shape = {num_unique};
    if (is_recompute) {
      NBLA_CHECK(outputs[0]->shape() == y_shape, error_code::value,
                 "Shape of y in recompute does not match in forward");
    }
    outputs[0]->reshape(y_shape, true);
    T *y = outputs[0]->cast_data_and_get_pointer<T>(ctx, true);
    for (int i = 0; i < num_unique; i++) {
      y[i] = x[indices_val[output_order[i]]];
    }
  } else {
    const T *x = inputs[0]->get_data_pointer<T>(ctx);
    const auto x_shape = inputs[0]->shape();
    Shape_t y_shape(x_shape);
    y_shape[axis] = num_unique;
    if (is_recompute) {
      NBLA_CHECK(outputs[0]->shape() == y_shape, error_code::value,
                 "Shape of y in recompute does not match in forward");
    }
    outputs[0]->reshape(y_shape, true);
    T *y = outputs[0]->cast_data_and_get_pointer<T>(ctx, true);

    Shape_t slice_shape(x_shape);
    slice_shape[axis] = 1;
    const auto slice_inner_size = ndi::inner_size(slice_shape, axis);
    const auto slice_outer_size = ndi::outer_size(slice_shape, axis);
    const auto src_axis_n = x_shape[axis];

    for (int i = 0; i < num_unique; i++) {
      auto y_offset = i * slice_inner_size;
      auto x_offset = indices_val[output_order[i]] * slice_inner_size;
      for (int j = 0; j < slice_outer_size; j++) {
        std::copy_n(x + x_offset, slice_inner_size, y + y_offset);
        y_offset += num_unique * slice_inner_size;
        x_offset += src_axis_n * slice_inner_size;
      }
    }
  }

  // Create output index
  int output_idx = 1;
  if (with_index) {
    Shape_t index_shape = {num_unique};
    if (is_recompute) {
      NBLA_CHECK(outputs[output_idx]->shape() == index_shape, error_code::value,
                 "Shape of indices in recompute does not match in forward");
    }
    outputs[output_idx]->reshape(index_shape, true);
    int *index = outputs[output_idx]->cast_data_and_get_pointer<int>(ctx, true);
    for (int i = 0; i < num_unique; i++) {
      index[i] = indices_val[output_order[i]];
    }
    output_idx++;
  }

  if (with_inverse) {
    vector<int> inverse_indices_table(num_unique);
    for (int i = 0; i < inverse_indices_table.size(); i++) {
      inverse_indices_table[output_order[i]] = i;
    }

    const Size_t axis_n = inverse_val.size();
    Shape_t inverse_shape = {axis_n};
    if (is_recompute) {
      NBLA_CHECK(
          outputs[output_idx]->shape() == inverse_shape, error_code::value,
          "Shape of inverse_indices in recompute does not match in forward");
    }
    outputs[output_idx]->reshape(inverse_shape, true);
    int *inverse =
        outputs[output_idx]->cast_data_and_get_pointer<int>(ctx, true);
    for (int i = 0; i < axis_n; i++) {
      inverse[i] = inverse_indices_table[inverse_val[i]];
    }
    output_idx++;
  }

  if (with_counts) {
    Shape_t counts_shape = {num_unique};
    if (is_recompute) {
      NBLA_CHECK(outputs[output_idx]->shape() == counts_shape,
                 error_code::value,
                 "Shape of counts in recompute does not match in forward");
    }
    outputs[output_idx]->reshape(counts_shape, true);
    int *counts =
        outputs[output_idx]->cast_data_and_get_pointer<int>(ctx, true);
    for (int i = 0; i < num_unique; i++) {
      counts[i] = counts_val[output_order[i]];
    }
  }
}

static void unique_preprocess(const Variables &inputs, const Context ctx,
                              bool flatten, int axis,
                              VariablePtr &reshaped_x_ptr) {
  // Reshape input to 2-D array.
  // Let's suppose that x shape is [X_0, X_1, X_2] and axis = 1.
  // If flatten is true, x is reshaped to [X_0 * X_1 * X_2, 1].
  // If flatten is false, x is transposed to [X_1, X_0 * X_2].
  const auto x_shape = inputs[0]->shape();
  if (flatten) {
    Shape_t reshaped_x_shape;
    reshaped_x_shape.push_back(std::accumulate(x_shape.cbegin(), x_shape.cend(),
                                               1, std::multiplies<int64_t>()));
    reshaped_x_shape.push_back(1);
    reshaped_x_ptr = inputs[0]->view(reshaped_x_shape);
  } else {
    // Transpose: [X_0, X_{axis}, X_2] -> [X_{axis}, X_0, X_2]
    Shape_t transposed_x_shape;
    vector<int> transposed_x_axes;
    transposed_x_shape.push_back(x_shape[axis]);
    transposed_x_axes.push_back(axis);
    for (Size_t i = 0; i < x_shape.size(); i++) {
      if (i != axis) {
        transposed_x_shape.push_back(x_shape[i]);
        transposed_x_axes.push_back(i);
      }
    }
    Variable transposed_x;
    auto f_transpose = create_Transpose(ctx, transposed_x_axes);
    f_transpose->setup(Variables{inputs[0]}, Variables{&transposed_x});
    f_transpose->forward(Variables{inputs[0]}, Variables{&transposed_x});

    // Reshape: [X_{axis}, X_0, X_2] -> [X_{axis}, X_0 * X_2]
    Shape_t reshaped_shape;
    reshaped_shape.push_back(transposed_x_shape[0]);
    reshaped_shape.push_back(std::accumulate(transposed_x_shape.cbegin() + 1,
                                             transposed_x_shape.cend(), 1,
                                             std::multiplies<int64_t>()));
    reshaped_x_ptr = transposed_x.view(reshaped_shape);
  }
}

template <typename T>
void Unique<T>::unique(const Variables &inputs, const Variables &outputs,
                       VariablePtr reshaped_x_ptr, bool is_recompute) {
  const auto x_shape = inputs[0]->shape();
  const T *x = reshaped_x_ptr->get_data_pointer<T>(this->ctx_);
  const auto reshaped_x_shape = reshaped_x_ptr->shape();
  const auto row_size = reshaped_x_shape[0];
  const auto col_size = reshaped_x_shape[1];

  vector<int> arg_indices(row_size);
  std::iota(arg_indices.begin(), arg_indices.end(), 0);
  std::stable_sort(arg_indices.begin(), arg_indices.end(),
                   [&](const int &a, const int &b) {
                     return std::lexicographical_compare(
                         x + a * col_size, x + (a + 1) * col_size,
                         x + b * col_size, x + (b + 1) * col_size);
                   });

  // Find unique value/slice and create indices, inverse_indices and counts when
  // sorted is true
  auto num_unique = 1;
  vector<int> indices_val{arg_indices[0]};
  vector<int> inverse_indices_val(row_size);
  inverse_indices_val[arg_indices[0]] = 0;
  vector<int> counts = {1};
  for (int i = 1; i < arg_indices.size(); i++) {
    const auto arg_current = arg_indices[i];
    const auto arg_previous = arg_indices[i - 1];
    if (!std::equal(x + arg_current * col_size,
                    x + (arg_current + 1) * col_size,
                    x + arg_previous * col_size)) {
      // Unique value is found
      indices_val.push_back(arg_current);
      arg_indices[num_unique] = arg_current;
      counts.push_back(1);
      inverse_indices_val[arg_current] = num_unique;
      num_unique++;
    } else {
      inverse_indices_val[arg_current] = num_unique - 1;
      counts[num_unique - 1]++;
    }
  }

  create_outputs<T>(this->ctx_, inputs, outputs, indices_val,
                    inverse_indices_val, counts, flatten_, axis_, sorted_,
                    with_index_, with_inverse_, with_counts_, is_recompute);
}

template <typename T>
void Unique<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  const auto auto_forward =
      SingletonManager::get<AutoForward>()->get_auto_forward();
  NBLA_CHECK(auto_forward, error_code::runtime,
             "Unique can be used only if auto_forward is true");

  const auto x_ndim = inputs[0]->ndim();
  refine_axis(axis_, x_ndim);

  VariablePtr reshaped_x_ptr;
  unique_preprocess(inputs, this->ctx_, this->flatten_, this->axis_,
                    reshaped_x_ptr);

  // Peform forward computation to determine the output shape.
  unique(inputs, outputs, reshaped_x_ptr);
}

template <typename T>
void Unique<T>::forward_impl(const Variables &inputs,
                             const Variables &outputs) {
  // Forward is done at setup_impl() because the output shape is calculated
  // during forward computation.
  const auto auto_forward =
      SingletonManager::get<AutoForward>()->get_auto_forward();
  NBLA_CHECK(auto_forward, error_code::runtime,
             "Unique can be used only if auto_forward is true");
}

template <typename T>
void Unique<T>::recompute_impl(const Variables &inputs,
                               const Variables &outputs) {
  VariablePtr reshaped_x_ptr;
  unique_preprocess(inputs, this->ctx_, this->flatten_, this->axis_,
                    reshaped_x_ptr);
  unique(inputs, outputs, reshaped_x_ptr, true);
}

template <typename T>
void Unique<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                              const vector<bool> &propagate_down,
                              const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }

  NBLA_ERROR(error_code::not_implemented,
             "Unique<T>::backward is currently not implemented.");
}
} // namespace nbla
