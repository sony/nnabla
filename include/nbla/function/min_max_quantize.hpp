// Copyright 2019,2020,2021 Sony Corporation.
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

// fixed_point_quantize.hpp
#ifndef NBLA_FUNCTION_MIN_MAX_QUANTIZE_HPP
#define NBLA_FUNCTION_MIN_MAX_QUANTIZE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(MinMaxQuantize, float, bool, bool, bool, float);

/** MinMaxQuantize quantizes values in integer representation.

Inputs:
- N-D array of input
- N-D array of minimum quantization range (modified during forward execution)
- N-D array of maximum quantization range (modified during forward execution)
- N-D array of minimum quantization level
- N-D array of maximum quantization level
execution)

@tparam T Data type for computation.

@param decay Decay rate for the exponential moving average.
@param x_min_max Use the min and max of x to compute quantization ranges.
@param ema Use the exponential moving average for the min and max quantization.
ranges.
@param ste_fine_grained Straight Through Estimator is fine-grained or not.

\ingroup FunctionImplGrp
 */
template <typename T>
class MinMaxQuantize : public BaseFunction<float, bool, bool, bool, float> {
protected:
  float decay_;
  bool x_min_max_;
  bool ema_;
  bool ste_fine_grained_;
  float eps_;
  shared_ptr<Function> identity_;
  shared_ptr<Function> round_;
  shared_ptr<Function> add2_;
  shared_ptr<Function> sub2_;
  shared_ptr<Function> mul2_;
  shared_ptr<Function> div2_;
  shared_ptr<Function> minimum2_;
  shared_ptr<Function> maximum2_;
  shared_ptr<Function> mul_scalar_;
  shared_ptr<Function> mul_scalar2_;
  shared_ptr<Function> min_;
  shared_ptr<Function> max_;
  shared_ptr<Function> broadcast_;
  shared_ptr<Function> greater_equal_;
  shared_ptr<Function> less_equal_;
  shared_ptr<Function> greater_;
  shared_ptr<Function> less_;
  shared_ptr<Function> sum_;
  VariablePtr scale_sptr_;

public:
  MinMaxQuantize(const Context &ctx, float decay, bool x_min_max, bool ema,
                 bool ste_fine_grained, float eps)
      : BaseFunction(ctx, decay, x_min_max, ema, ste_fine_grained, eps),
        decay_(decay), x_min_max_(x_min_max), ema_(ema),
        ste_fine_grained_(ste_fine_grained), eps_(eps) {}
  virtual ~MinMaxQuantize() {}
  virtual shared_ptr<Function> copy() const {
    return create_MinMaxQuantize(ctx_, decay_, x_min_max_, ema_,
                                 ste_fine_grained_, eps_);
  }
  virtual int min_inputs() { return 5; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>(),
                          get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "MinMaxQuantize"; }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  NBLA_API virtual void nudge_range(Variable *qr_min, Variable *qr_max);
  NBLA_API virtual void nudge_qr_min_max(Variable *qr_min, Variable *qr_max,
                                         Variable *ql_min, Variable *ql_max,
                                         Variable *scale,
                                         Variable *qr_min_nudged,
                                         Variable *qr_max_nudged);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (i == 0 || i == 1 || i == 2) {
      return true;
    }
    return false;
  }
};
}
#endif
