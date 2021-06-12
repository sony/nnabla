// Copyright 2021 Sony Corporation.
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
  vector<int> tn2bn_transpose_axes_, bn2tn_transpose_axes_;
  Shape_t transposed_shape_, bn_shape;
  shared_ptr<Function> tn2bn_transpose_, bn2tn_transpose_;

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
    tn2bn_transpose_axes_.reserve(outer_axes.size() + inner_axes.size());
    tn2bn_transpose_axes_.insert(tn2bn_transpose_axes_.end(),
                                 outer_axes.begin(), outer_axes.end());
    tn2bn_transpose_axes_.insert(tn2bn_transpose_axes_.end(),
                                 inner_axes.begin(), inner_axes.end());

    // tranposed_shape
    for (int i = 0; i < ndim; i++) {
      transposed_shape_.push_back(in_shape[tn2bn_transpose_axes_[i]]);
    }

    // bn_shape
    int64_t reduced_inner_size = 1;
    for (size_t i = outer_axes.size(); i < transposed_shape_.size(); i++) {
      reduced_inner_size *= transposed_shape_[i];
    }
    for (size_t i = 0; i < outer_axes.size(); i++) {
      bn_shape.push_back(transposed_shape_[i]);
    }
    bn_shape.push_back(reduced_inner_size);
    // ndim of batch_norm input x muse be greater than or equal to 2.
    if (bn_shape.size() == 1) {
      bn_shape = {1, bn_shape[0]};
    }

    // inv_transpose_axes = argsort(transpose_axes)
    vector<pair<int, int>> tn2bn_transpose_axes_with_index;
    for (int i = 0; i < ndim; i++) {
      tn2bn_transpose_axes_with_index.push_back({tn2bn_transpose_axes_[i], i});
    }
    std::sort(tn2bn_transpose_axes_with_index.begin(),
              tn2bn_transpose_axes_with_index.end());
    for (int i = 0; i < ndim; i++) {
      bn2tn_transpose_axes_.push_back(
          tn2bn_transpose_axes_with_index[i].second);
    }

    // functions
    tn2bn_transpose_ = create_Transpose(ctx, tn2bn_transpose_axes_);
    bn2tn_transpose_ = create_Transpose(ctx, bn2tn_transpose_axes_);
  }

  // tensor_norm to batch_norm
  void tn2bn(Variable *from, Variable *to) {
    nbla::execute(tn2bn_transpose_, {from}, {to});
    to->reshape(bn_shape, false);
  }

  void tn2bn_backward(Variable *from, Variable *to, bool propagate_down,
                      bool accum) {
    to->reshape(transposed_shape_, false);
    nbla::backward(tn2bn_transpose_, {from}, {to}, {propagate_down}, {accum});
  }

  // batch_norm to tensor_norm adapter
  void bn2tn(Variable *from, Variable *to) {
    from->reshape(transposed_shape_, false);
    nbla::execute(bn2tn_transpose_, {from}, {to});
  }

  void bn2tn_backward(Variable *from, Variable *to, bool propagate_down,
                      bool accum) {
    // this reshape is needed because bn2tn is skipped in tensor_norm backward.
    from->reshape(transposed_shape_, false);
    nbla::backward(bn2tn_transpose_, {from}, {to}, {propagate_down}, {accum});
    from->reshape(bn_shape, false);
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
  bool need_adapter_;
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
  NBLA_API virtual void setup_batch_norm(const Variables &inputs,
                                         const Variables &outputs);
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_with_adapter(const Variables &inputs,
                                             const Variables &outputs);
  NBLA_API virtual void forward_without_adapter(const Variables &inputs,
                                                const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void
  backward_with_adapter(const Variables &inputs, const Variables &outputs,
                        const vector<bool> &propagate_down,
                        const vector<bool> &accum);
  NBLA_API virtual void
  backward_without_adapter(const Variables &inputs, const Variables &outputs,
                           const vector<bool> &propagate_down,
                           const vector<bool> &accum);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (need_adapter_) {
      if (i == 0) {
        return true;
      }
      if (j == 0) {
        return true;
      }
      if (i == j) {
        return true;
      }
    } else {
      if (i == 0) {
        if (j == 0 || j == gamma_idx_) {
          return true;
        }
      }
      if (i == gamma_idx_) {
        if (j == 0) {
          return true;
        }
      }
    }
    return false;
  }
};
}
#endif
