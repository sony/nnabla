// Copyright 2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_FUSED_CONVOLUTION_HPP
#define NBLA_FUNCTION_FUSED_CONVOLUTION_HPP

#include <nbla/computation_graph/variable.hpp>
#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <unordered_map>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(FusedConvolution, int, const vector<int> &,
                              const vector<int> &, const vector<int> &, int,
                              bool, float, float, bool, const string &,
                              const vector<float> &);

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T>
class FusedConvolution
    : public BaseFunction<int, const vector<int> &, const vector<int> &,
                          const vector<int> &, int, bool, float, float, bool,
                          const string &, const vector<float> &> {
protected:
  int base_axis_;
  const vector<int> pad_;
  const vector<int> stride_;
  const vector<int> dilation_;
  int group_;
  bool channel_last_;
  float decay_rate_;
  float eps_;
  bool batch_stat_;
  const string nonlinearity_;
  const vector<float> nonlinearity_args_;

public:
  // Name of input variables
  typedef int InName;
  const InName X = 0, WEIGHT = 1, BIAS = 2, BETA = 3, GAMMA = 4, MEAN = 5,
               VARIANCE = 6, Z = 7;

public:
  FusedConvolution(const Context &ctx, int base_axis, const vector<int> &pad,
                   const vector<int> &stride, const vector<int> &dilation,
                   int group, bool channel_last, float decay_rate, float eps,
                   bool batch_stat, const string &nonlinearity,
                   const vector<float> &nonlinearity_args)
      : BaseFunction(ctx, base_axis, pad, stride, dilation, group, channel_last,
                     decay_rate, eps, batch_stat, nonlinearity,
                     nonlinearity_args),
        base_axis_(base_axis), pad_(pad), stride_(stride), dilation_(dilation),
        group_(group), channel_last_(channel_last), decay_rate_(decay_rate),
        eps_(eps), batch_stat_(batch_stat), nonlinearity_(nonlinearity),
        nonlinearity_args_(nonlinearity_args) {}
  virtual ~FusedConvolution() {}
  virtual shared_ptr<Function> copy() const {
    return create_FusedConvolution(
        ctx_, base_axis_, pad_, stride_, dilation_, group_, channel_last_,
        decay_rate_, eps_, batch_stat_, nonlinearity_, nonlinearity_args_);
  }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>(),
                          get_dtype<T>(), get_dtype<T>(), get_dtype<T>(),
                          get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "FusedConvolution"; }
  virtual bool grad_depends_output_data(int i, int o) const {
    if (nonlinearity_ == "relu") {
      return true;
    } else if (nonlinearity_ == "sigmoid") {
      return true;
    } else if (nonlinearity_ == "tanh") {
      return true;
    } else if (nonlinearity_ == "leaky_relu") {
      return true;
    }
    return false;
  }

protected:
  /**
    Get pointers of variables from a vector. A pointer to Variable will be set
    to a referenced variable in the arguments if specified. Otherwise, nullptr
    will be set.
  */
  NBLA_API void get_optional_input_pointers(const Variables &inputs);
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void recompute_impl(const Variables &inputs,
                                       const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);

  std::unordered_map<InName, std::pair<int, Variable *>> input_variables_;
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (input_variables_.find(GAMMA) != input_variables_.end())
      return true;
    if (i == X && j == WEIGHT) {
      return true;
    }
    if (i == WEIGHT && j == X) {
      return true;
    }
    return false;
  }
  virtual bool overwrite_input_data_in_forward_impl(int i) const {
    // mean, variance
    if (i == 4 || i == 5) {
      return true;
    }
    return false;
  }

private:
  // Members only used in a naive implementation with composite
  std::unordered_map<InName, CgVariablePtr> input_cg_variables_;
  CgVariablePtr last_output_cg_variable_;
  bool reset_cg_variables(const Variables &inputs, const Variables &outputs);
};
}
#endif
