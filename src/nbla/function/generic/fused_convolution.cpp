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
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/function/fused_convolution.hpp>
#include <nbla/functions.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(FusedConvolution, int, const vector<int> &,
                              const vector<int> &, const vector<int> &, int,
                              bool, float, float, bool, const string &,
                              const vector<float> &);

template <typename T>
void FusedConvolution<T>::get_optional_input_pointers(const Variables &inputs) {
  input_variables_[X] = {0, inputs[0]};
  input_variables_[WEIGHT] = {1, inputs[1]};
  auto set_bn_ptrs = [&]() {
    input_variables_[BETA] = {2, inputs[2]};
    input_variables_[GAMMA] = {3, inputs[3]};
    input_variables_[MEAN] = {4, inputs[4]};
    input_variables_[VARIANCE] = {5, inputs[5]};
  };

  // Combination is quite complex. Writing down for all patterns
  switch (inputs.size()) {
  case 2:
    break;
  case 3:
    // Assume bias vector is one dimensional vector, while z has a spatial
    // structure.
    if (inputs[2]->ndim() == 1) {
      input_variables_[BIAS] = {2, inputs[2]};
    } else {
      input_variables_[Z] = {2, inputs[2]};
    }
    break;
  case 4:
    // No BN.
    input_variables_[BIAS] = {2, inputs[2]};
    input_variables_[Z] = {3, inputs[3]};
    break;
  case 6:
    // No bias nor z
    set_bn_ptrs();
    break;
  case 7: {
    // Assume bias vector is one dimensional vector, while bn args and z have
    // spatial dimensions.
    NBLA_CHECK(inputs[2]->ndim() != 1, error_code::value,
               "Wrong input shape. It seems that you pass convolution bias "
               "and batch normalization inputs at the same time (prohibited).");
    // No bias nor z
    set_bn_ptrs();
    input_variables_[Z] = {6, inputs[6]};
    break;
  }
  default:
    NBLA_ERROR(error_code::value,
               "Unknown error. Wrong number of arguments are specified.");
  }
}

namespace {
CgVariablePtr create_cgvariable_from_variable(Variable *var, bool need_grad) {
  auto cg_var = make_shared<CgVariable>(var->shape(), need_grad);
  cg_var->variable()->set_data(var->data());
  cg_var->variable()->set_grad(var->grad());
  return cg_var;
}

bool reset_cgvariable(CgVariablePtr cg_var, Variable *var) {
  bool ret = false;
  // std::cout << cg_var->variable()->data().get() << " ? " << var->data().get()
  //           << std::endl;
  if (cg_var->variable()->data() != var->data()) {
    cg_var->variable()->set_data(var->data());
    ret = true;
  }
  // std::cout << cg_var->variable()->grad().get() << " ? " << var->grad().get()
  //           << std::endl;
  if (cg_var->variable()->grad() != var->grad()) {
    cg_var->variable()->set_grad(var->grad());
    ret = true;
  }
  return ret;
}
}

template <typename T>
bool FusedConvolution<T>::reset_cg_variables(const Variables &inputs,
                                             const Variables &outputs) {
  auto get_index = [this](InName name) {
    return this->input_variables_[name].first;
  };
  bool ret = false;
  ret |= reset_cgvariable(input_cg_variables_[X], inputs[get_index(X)]);
  ret |=
      reset_cgvariable(input_cg_variables_[WEIGHT], inputs[get_index(WEIGHT)]);
  if (input_cg_variables_[BIAS]) {
    ret |= reset_cgvariable(input_cg_variables_[BIAS], inputs[get_index(BIAS)]);
  }
  if (input_cg_variables_[BETA]) {
    ret |= reset_cgvariable(input_cg_variables_[BETA], inputs[get_index(BETA)]);
    ret |=
        reset_cgvariable(input_cg_variables_[GAMMA], inputs[get_index(GAMMA)]);
    ret |= reset_cgvariable(input_cg_variables_[MEAN], inputs[get_index(MEAN)]);
    ret |= reset_cgvariable(input_cg_variables_[VARIANCE],
                            inputs[get_index(VARIANCE)]);
  }
  if (input_cg_variables_[Z]) {
    ret |= reset_cgvariable(input_cg_variables_[Z], inputs[get_index(Z)]);
  }
  return ret;
}

template <typename T>
void FusedConvolution<T>::setup_impl(const Variables &inputs,
                                     const Variables &outputs) {

  this->get_optional_input_pointers(inputs);

  // ----------------------------------------------------------------
  // Convolution
  // ----------------------------------------------------------------
  auto cg_x = create_cgvariable_from_variable(input_variables_[X].second, true);
  auto cg_weight =
      create_cgvariable_from_variable(input_variables_[WEIGHT].second, true);
  input_cg_variables_[X] = cg_x;
  input_cg_variables_[WEIGHT] = cg_weight;
  CgVariablePtr cg_bias;
  if (input_variables_[BIAS].second) {
    cg_bias =
        create_cgvariable_from_variable(input_variables_[BIAS].second, true);
    input_cg_variables_[BIAS] = cg_bias;
  }
  auto last_out =
      functions::convolution(ctx_, cg_x, cg_weight, cg_bias, base_axis_, pad_,
                             stride_, dilation_, group_, channel_last_)[0];

  // ----------------------------------------------------------------
  // BatchNormalization
  // ----------------------------------------------------------------
  if (input_variables_[BETA].second) {
    auto cg_beta =
        create_cgvariable_from_variable(input_variables_[BETA].second, true);
    auto cg_gamma =
        create_cgvariable_from_variable(input_variables_[GAMMA].second, true);
    auto cg_mean =
        create_cgvariable_from_variable(input_variables_[MEAN].second, true);
    auto cg_variance = create_cgvariable_from_variable(
        input_variables_[VARIANCE].second, true);
    input_cg_variables_[BETA] = cg_beta;
    input_cg_variables_[GAMMA] = cg_gamma;
    input_cg_variables_[MEAN] = cg_mean;
    input_cg_variables_[VARIANCE] = cg_variance;
    vector<int> axes{channel_last_ ? (int)(inputs[0]->ndim() - 1) : base_axis_};
    last_out = functions::batch_normalization(
        ctx_, last_out, cg_beta, cg_gamma, cg_mean, cg_variance, axes,
        decay_rate_, eps_, batch_stat_, false /* no_scale */,
        false /* no_bias */)[0];
  }

  // ----------------------------------------------------------------
  // Add2
  // ----------------------------------------------------------------
  if (input_variables_[Z].second) {
    auto cg_z =
        create_cgvariable_from_variable(input_variables_[Z].second, true);
    input_cg_variables_[Z] = cg_z;
    last_out = functions::add2(ctx_, last_out, cg_z, true /* inplace */)[0];
  }

  // ----------------------------------------------------------------
  // Activation
  // ----------------------------------------------------------------
  size_t num_args = nonlinearity_args_.size();
  if (nonlinearity_ == "identity" || nonlinearity_.empty()) {
  } else if (nonlinearity_ == "relu") {
    last_out = functions::relu(ctx_, last_out, true)[0];
  } else if (nonlinearity_ == "sigmoid") {
    last_out = functions::sigmoid(ctx_, last_out)[0];
  } else if (nonlinearity_ == "tanh") {
    last_out = functions::tanh(ctx_, last_out)[0];
  } else if (nonlinearity_ == "leaky_relu") {
    NBLA_CHECK(num_args == 1, error_code::value,
               "LeakyReLU requires 1 arguments in nonlinearity_args (alpha).");
    last_out =
        functions::leaky_relu(ctx_, last_out, nonlinearity_args_[0], true)[0];
  } else if (nonlinearity_ == "elu") {
    NBLA_CHECK(num_args == 1, error_code::value,
               "ELU requires 1 arguments in nonlinearity_args (alpha).");
    last_out = functions::elu(ctx_, last_out, nonlinearity_args_[0])[0];
  } else if (nonlinearity_ == "relu6") {
    last_out = functions::relu6(ctx_, last_out)[0];
  } else {
    NBLA_ERROR(error_code::not_implemented,
               "Not implemented activation type %s", nonlinearity_.c_str());
  }
  // Replace the output variable in the last CgVariable with outputs[0].

  outputs[0]->reshape(last_out->variable()->shape(), true);
  last_out->variable()->set_data(outputs[0]->data());
  last_out->variable()->set_grad(outputs[0]->grad());
  // Call all setup again to ensure inplaced variable is refer to the correct
  // array.
  std::unordered_set<CgFunctionPtr> fclosed;
  last_out->visit_function_recursive(last_out->parent(), fclosed,
                                     [](CgFunctionPtr fn) { fn->setup(); });
  this->last_output_cg_variable_ = last_out;
}

template <typename T>
void FusedConvolution<T>::forward_impl(const Variables &inputs,
                                       const Variables &outputs) {
  reset_cg_variables(inputs, outputs);
  bool clear_buffer =
      SingletonManager::get<GlobalClearBufferState>()->clear_buffer();
  last_output_cg_variable_->forward(clear_buffer, false);
}

template <typename T>
void FusedConvolution<T>::backward_impl(const Variables &inputs,
                                        const Variables &outputs,
                                        const vector<bool> &propagate_down,
                                        const vector<bool> &accum) {
  reset_cg_variables(inputs, outputs);
  auto get_index = [this](InName name) {
    return this->input_variables_[name].first;
  };
  input_cg_variables_[X]->set_need_grad(propagate_down[get_index(X)]);
  input_cg_variables_[WEIGHT]->set_need_grad(propagate_down[get_index(WEIGHT)]);
  if (input_cg_variables_[BIAS]) {
    input_cg_variables_[BIAS]->set_need_grad(propagate_down[get_index(BIAS)]);
  }
  if (input_cg_variables_[BETA]) {
    input_cg_variables_[BETA]->set_need_grad(propagate_down[get_index(BETA)]);
    input_cg_variables_[GAMMA]->set_need_grad(propagate_down[get_index(GAMMA)]);
    input_cg_variables_[MEAN]->set_need_grad(propagate_down[get_index(MEAN)]);
    input_cg_variables_[VARIANCE]->set_need_grad(
        propagate_down[get_index(VARIANCE)]);
  }
  if (input_cg_variables_[Z]) {
    input_cg_variables_[Z]->set_need_grad(propagate_down[get_index(Z)]);
  }

  // Propagate need_grad states
  std::unordered_set<CgFunctionPtr> fclosed;
  last_output_cg_variable_->visit_function_recursive(
      last_output_cg_variable_->parent(), fclosed, [](CgFunctionPtr fn) {});

  last_output_cg_variable_->backward(outputs[0]->grad(), true);
}
}
