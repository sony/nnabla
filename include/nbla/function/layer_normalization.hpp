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

#ifndef NBLA_FUNCTION_LAYER_NORMALIZATION_HPP
#define NBLA_FUNCTION_LAYER_NORMALIZATION_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(LayerNormalization, const vector<int> &, float,
                              bool, bool);

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T>
class LayerNormalization
    : public BaseFunction<const vector<int> &, float, bool, bool> {
protected:
  vector<int> batch_axis_;
  float eps_;
  bool no_scale_, no_bias_;

  int beta_idx_, gamma_idx_;
  bool output_stat_;
  Shape_t tn_shape_;
  shared_ptr<Function> f_tensor_norm_;
  shared_ptr<Function> f_mul2_;
  shared_ptr<Function> f_add2_;
  shared_ptr<Function> f_sub2_;

public:
  LayerNormalization(const Context &ctx, const vector<int> &batch_axis,
                     float eps, bool no_scale, bool no_bias)
      : BaseFunction(ctx, batch_axis, eps, no_scale, no_bias),
        batch_axis_(batch_axis), eps_(eps), no_scale_(no_scale),
        no_bias_(no_bias) {}
  virtual ~LayerNormalization() {}
  virtual shared_ptr<Function> copy() const {
    return create_LayerNormalization(ctx_, batch_axis_, eps_, no_scale_,
                                     no_bias_);
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
  virtual string name() { return "LayerNormalization"; }
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
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (i == 0) {
      return true;
    }
    if (j == 0) {
      return true;
    }
    if (i == j) {
      return true;
    }
    return false;
  }
};
}
#endif
