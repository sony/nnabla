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

#ifndef NBLA_FUNCTION_TENSOR_NORMALIZATION_HPP
#define NBLA_FUNCTION_TENSOR_NORMALIZATION_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <nbla/function/batch_normalization.hpp>
#include <nbla/function/reshape.hpp>
#include <nbla/function/transpose.hpp>
#include <nbla/imperative.hpp>

#include <numeric>

namespace nbla {

class BatchNormalizationInOutAdapter {
  vector<int> transpose_axes_, inv_transpose_axes_;
  Shape_t transposed_shape_, bn_input_shape_;
  shared_ptr<Function> pre_op_transpose_, post_op_transpose_;

public:
  BatchNormalizationInOutAdapter(const Context &ctx, int ndim,
                                 const Shape_t &in_shape,
                                 const vector<int> &axes) {
    // inner_axes = sort(axes)
    auto inner_axes = axes;
    std::sort(inner_axes.begin(), inner_axes.end());

    // outer_axes = sort(range(ndim) / inner_axes)
    vector<int> outer_axes, range_ndim(ndim);
    std::iota(range_ndim.begin(), range_ndim.end(), 0);
    std::set_difference(range_ndim.begin(), range_ndim.end(),
                        inner_axes.begin(), inner_axes.end(),
                        std::inserter(outer_axes, outer_axes.begin()));
    std::sort(outer_axes.begin(), outer_axes.end());

    // transpose_axes = outer_axes + inner_axes
    transpose_axes_.reserve(outer_axes.size() + inner_axes.size());
    transpose_axes_.insert(transpose_axes_.end(), outer_axes.begin(),
                           outer_axes.end());
    transpose_axes_.insert(transpose_axes_.end(), inner_axes.begin(),
                           inner_axes.end());

    // tranposed_shape
    for (int i = 0; i < ndim; i++) {
      transposed_shape_.push_back(in_shape[transpose_axes_[i]]);
    }

    // bn_input_shape
    int64_t reduced_inner_size = 1;
    for (int i = outer_axes.size(); i < transposed_shape_.size(); i++) {
      reduced_inner_size *= transposed_shape_[i];
    }
    for (int i = 0; i < outer_axes.size(); i++) {
      bn_input_shape_.push_back(transposed_shape_[i]);
    }
    bn_input_shape_.push_back(reduced_inner_size);
    // ndim of batch_norm input x muse be greater than or equal to 2.
    if (bn_input_shape_.size() == 1) {
      bn_input_shape_ = {1, bn_input_shape_[0]};
    }

    // inv_transpose_axes = argsort(transpose_axes)
    vector<pair<int, int>> transpose_axes_with_index;
    for (int i = 0; i < ndim; i++) {
      transpose_axes_with_index.push_back({transpose_axes_[i], i});
    }
    std::sort(transpose_axes_with_index.begin(),
              transpose_axes_with_index.end());
    for (int i = 0; i < ndim; i++) {
      inv_transpose_axes_.push_back(transpose_axes_with_index[i].second);
    }

    // functions
    pre_op_transpose_ = create_Transpose(ctx, transpose_axes_);
    post_op_transpose_ = create_Transpose(ctx, inv_transpose_axes_);
  }

  // batch_norm inputs adapter
  void pre_op(Variable *x, Variable *y) {
    nbla::execute(pre_op_transpose_, {x}, {y});
    y->reshape(bn_input_shape_, false);
  }

  void pre_op_backward(Variable *x, Variable *y, bool propagate_down,
                       bool accum) {
    y->reshape(transposed_shape_, false);
    nbla::backward(pre_op_transpose_, {x}, {y}, {propagate_down}, {accum});
  }

  // batch_norm outputs adapter
  void post_op(Variable *x, Variable *y) {
    x->reshape(transposed_shape_, false);
    nbla::execute(post_op_transpose_, {x}, {y});
  }

  void post_op_backward(Variable *x, Variable *y, bool propagate_down,
                        bool accum) {
    nbla::backward(post_op_transpose_, {x}, {y}, {propagate_down}, {accum});
    x->reshape(bn_input_shape_, false);
  }
};

NBLA_REGISTER_FUNCTION_HEADER(TensorNormalization, const vector<int> &, float,
                              bool, bool);

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T>
class TensorNormalization
    : public BaseFunction<const vector<int> &, float, bool, bool> {
protected:
  const vector<int> axes_;
  float eps_;
  bool no_scale_, no_bias_;

  int beta_idx_, gamma_idx_;
  bool output_stat_;
  Shape_t bn_param_shape_;
  std::unique_ptr<BatchNormalizationInOutAdapter> bn_in_adapter_,
      bn_param_adapter_;
  shared_ptr<Function> f_batch_norm_;

public:
  TensorNormalization(const Context &ctx, const vector<int> &axes, float eps,
                      bool no_scale, bool no_bias)
      : BaseFunction(ctx, axes, eps, no_scale, no_bias), axes_(axes), eps_(eps),
        no_scale_(no_scale), no_bias_(no_bias) {}
  virtual ~TensorNormalization() {}
  virtual shared_ptr<Function> copy() const {
    return create_TensorNormalization(ctx_, axes_, eps_, no_scale_, no_bias_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "TensorNormalization"; }
  virtual bool grad_depends_output_data(int i, int o) const {
    // Gradient computation always requires output mean and var.
    return o > 0;
  }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
}
#endif
