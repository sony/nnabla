// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

#include <nbla/function.hpp>

#include <algorithm>
#include <memory>

namespace nbla {

Function::Function(const Context &ctx) : ctx_(ctx), fall_back_func_(nullptr) {}

Function::~Function() {}

void Function::setup(const Variables &inputs, const Variables &outputs) {
  called_setup_ = true;

  if (fall_back_func_) {
    // Fall back to the specified Function.
    fall_back_func_->setup(inputs, outputs);
    return;
  }

  //
  // Check if the context array_class matches any of the functions allowed
  // array classes. If not, choose the 0-th array class as default.
  //
  auto i = allowed_array_classes().size();
  while ((i > 0) && (ctx_.array_class != allowed_array_classes().at(--i)))
    ;
  ctx_.set_array_class(allowed_array_classes().at(i));

  //
  // Check the number of inputs and outputs is at least the minimum required.
  //
  auto min_inputs = static_cast<Variables::size_type>(this->min_inputs());
  auto min_outputs = static_cast<Variables::size_type>(this->min_outputs());
  NBLA_CHECK(min_inputs <= inputs.size(), error_code::value,
             "%s needs at least %d inputs (given %d). ", this->name().c_str(),
             this->min_inputs(), inputs.size());
  NBLA_CHECK(min_outputs <= outputs.size(), error_code::value,
             "%s needs at least %d outputs (given %d). ", this->name().c_str(),
             this->min_outputs(), outputs.size());

  this->setup_impl(inputs, outputs);

  if (fall_back_func_) {
    return;
  }

  // Memorize shapes
  in_shapes.clear();
  out_shapes.clear();
  for (Variables::size_type i = 0; i < inputs.size(); ++i) {
    in_shapes.push_back(make_shared<Shape_t>(inputs[i]->shape()));
  }
  for (Variables::size_type i = 0; i < outputs.size(); ++i) {
    out_shapes.push_back(make_shared<Shape_t>(outputs[i]->shape()));
  }
}

static void check_shapes(Function *function, const Variables &inputs,
                         const Variables &outputs,
                         const vector<shared_ptr<Shape_t>> &in_shapes,
                         const vector<shared_ptr<Shape_t>> &out_shapes) {
  NBLA_CHECK(inputs.size() == in_shapes.size(), error_code::value,
             "Num of inputs has been changed since setup is called in %s. "
             "Given: %d != previously: %d. ",
             function->name().c_str(), inputs.size(), in_shapes.size());
  NBLA_CHECK(outputs.size() == out_shapes.size(), error_code::value,
             "Num of outputs has been changed since setup is called in %s. "
             "Given: %d != previously: %d. ",
             function->name().c_str(), outputs.size(), out_shapes.size());
  for (Variables::size_type i = 0; i < inputs.size(); ++i) {
    NBLA_CHECK(*in_shapes[i] == inputs[i]->shape(), error_code::value,
               "Inconsistent shape in input %d of %s. "
               "Setup: (%s) != Given: (%s).",
               i, function->name().c_str(),
               string_join(*(in_shapes[i]), string(", ")).c_str(),
               string_join(inputs[i]->shape(), string(", ")).c_str());
  }
  for (Variables::size_type i = 0; i < outputs.size(); ++i) {
    NBLA_CHECK(*out_shapes[i] == outputs[i]->shape(), error_code::value,
               "Inconsistent shape in output %d of %s. "
               "Setup: (%s) != Given: (%s).",
               i, function->name().c_str(),
               string_join(*(out_shapes[i]), string(", ")).c_str(),
               string_join(outputs[i]->shape(), string(", ")).c_str());
  }
}

void Function::forward(const Variables &inputs, const Variables &outputs) {
  if (fall_back_func_) {
    // Fall back to the specified Function.
    fall_back_func_->forward(inputs, outputs);
    return;
  }
  check_shapes(this, inputs, outputs, in_shapes, out_shapes);
  this->forward_impl(inputs, outputs);
}

void Function::backward(const Variables &inputs, const Variables &outputs,
                        const vector<bool> &propagate_down,
                        const vector<bool> &accum) {
  if (fall_back_func_) {
    // Fall back to the specified Function.
    fall_back_func_->backward(inputs, outputs, propagate_down, accum);
    return;
  }
  check_shapes(this, inputs, outputs, in_shapes, out_shapes);
  // Always zero-ing gradient buffer when accum is false.
  // NNabla's backward implementation takes an accum flag for each input
  // variable. An accum flag is automatically determined by our graph engine.
  // Suppose we have a Variable, and it is used twice in two difference
  // Functions. The input variable of two different functions is responsible for
  // storing the gradients from the functions, which are summed up. A simpler
  // implementation to achieve this is to insert a split function that splits a
  // variable to two and that is responsible for summing up the gradient signals
  // from two outputs. However, in NNabla, to reduce computation and memory
  // overhead, this is achieved in each function by the `accum` flag. In the
  // first of two functions, `accum` is set as false by the graph engine, and
  // the backward signal is "written" to the gradient buffer of the input
  // variable. In the second function, `accum` is set as true, and the backward
  // signal is "accumulated" to the buffer.
  // In some functions, gradient is not computed (not defined), and a developer
  // might think zeros should be propagated to the inputs gradient, and a
  // developer probably does nothing in backward implementation. However, if
  // `accum` is false and grad is initialized as following, the gradient buffer
  // is not initialized (i.e. values in buffer are undefined), and propagated to
  // predecessor functions, which is a hazardous behavior.
  // To make it safer, gradients are initialized here as zeros when `accum` is
  // false.
  // Note that this does not impose overhead by explicitly calling zero()
  // function when write_only access to variable grad is properly used, because
  // zero() function is lazily evaluated and write_only option in
  // Variable::cast* function resets all lazy-evaluation flags before getting an
  // array instance.
  if (!this->prohibit_zero_input_grad()) {
    for (Variables::size_type i = 0; i < inputs.size(); i++) {
      if (propagate_down[i] && !accum[i]) {
        inputs[i]->grad()->zero();
      }
    }
  }
  // Calling the sub-class implementation of backward.
  this->backward_impl(inputs, outputs, propagate_down, accum);
}

void Function::setup_recompute(const Variables &inputs,
                               const Variables &outputs) {
  this->setup_recompute_impl(inputs, outputs);
  called_setup_recompute_ = true;
}

void Function::recompute(const Variables &inputs, const Variables &outputs) {
  // Check whether `setup_recompute` is called correctly.
  for (Variables::size_type o = 0; o < outputs.size(); o++) {
    if (need_setup_recompute(o)) {
      NBLA_CHECK(called_setup_recompute_, error_code::runtime,
                 "%s needs to execute `setup_recompute()` before calling "
                 "`recompute()`.",
                 name().c_str(), name().c_str());
    }
  }

  if (fall_back_func_) {
    // Fall back to the specified Function.
    fall_back_func_->recompute(inputs, outputs);
    return;
  }
  this->recompute_impl(inputs, outputs);
  called_setup_recompute_ = false;
}

//
// Set a new function input mask where true values indicate that the graph
// preceeding the corresponding function input shall be evaluated by the
// graph engine, and false values disable that graph leading to the input.
//
// Functions that handle inactive inputs must have cg_input_mask set to the
// same number of truth values as their inputs count. For all other functions
// the cg_input_mask vector will be empty.
//
void Function::set_active_input_mask(const vector<bool> &mask) {
  NBLA_CHECK(cg_input_mask.size() > 0, error_code::value,
             "%s function does not allow to deactivate inputs.",
             this->name().c_str());
  NBLA_CHECK(mask.size() == cg_input_mask.size(), error_code::value,
             "Mask size must match the number of %s function inputs.",
             this->name().c_str());
  cg_input_mask = mask;
}

//
// Query the input mask value for the i-th output. This is used by the graph
// engine to determine if the graph leading to this input must be considered.
//
// The cg_input_mask vector is either empty (for functions not handling
// inactive inputs) or the same size as the number of inputs.
//
bool Function::is_active_input(int i) const {
  if (i < this->cg_input_mask.size()) {
    return cg_input_mask[i];
  }
  return true;
}

Context Function::context() const { return ctx_; }
}
