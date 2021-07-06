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

#ifndef __NBLA_FUNCTION_CALLBACK_HPP__
#define __NBLA_FUNCTION_CALLBACK_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

using std::string;

/** A callback Function.

The callback functions of setup_impl, forward_impl and backward_mpl registered
at initialization are called.

 */
class Callback : public BaseFunction<> {

public:
  typedef std::function<void(void *, const Variables &inputs,
                             const Variables &outputs)>
      setup_callback_type;
  typedef std::function<void(void *, const Variables &inputs,
                             const Variables &outputs)>
      forward_callback_type;
  typedef std::function<void(
      void *, const Variables &inputs, const Variables &outputs,
      const vector<bool> &propagate_down, const vector<bool> &accum)>
      backward_callback_type;
  typedef std::function<void(void *)> cleanup_callback_type;

  typedef std::function<bool(void *, int, int)>
      grad_depends_output_data_callback_type;
  typedef std::function<bool(void *, int, int)>
      grad_depends_input_data_callback_type;

private:
  void *obj_;
  int min_outputs_;
  setup_callback_type setup_callback_;
  forward_callback_type forward_callback_;
  backward_callback_type backward_callback_;
  cleanup_callback_type cleanup_callback_;
  grad_depends_output_data_callback_type grad_depends_output_data_callback_ =
      nullptr;
  grad_depends_input_data_callback_type grad_depends_input_data_callback_ =
      nullptr;

public:
  Callback(const Context &ctx, void *obj, int min_outputs,
           setup_callback_type s, forward_callback_type f,
           backward_callback_type b, cleanup_callback_type c)
      : BaseFunction(ctx), obj_(obj), min_outputs_(min_outputs),
        setup_callback_(s), forward_callback_(f), backward_callback_(b),
        cleanup_callback_(c) {}
  Callback(const Context &ctx, void *obj, int min_outputs,
           setup_callback_type s, forward_callback_type f,
           backward_callback_type b, cleanup_callback_type c,
           grad_depends_output_data_callback_type go,
           grad_depends_input_data_callback_type gi)
      : BaseFunction(ctx), obj_(obj), min_outputs_(min_outputs),
        setup_callback_(s), forward_callback_(f), backward_callback_(b),
        cleanup_callback_(c), grad_depends_output_data_callback_(go),
        grad_depends_input_data_callback_(gi) {}
  virtual ~Callback() { cleanup_callback_(obj_); }
  virtual shared_ptr<Function> copy() const {
    return std::make_shared<Callback>(
        ctx_, obj_, min_outputs_, setup_callback_, forward_callback_,
        backward_callback_, cleanup_callback_,
        grad_depends_output_data_callback_, grad_depends_input_data_callback_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return min_outputs_; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<float>()};
  }
  virtual vector<dtypes> out_types() {
    return vector<dtypes>{get_dtype<float>()};
  }
  virtual string name() { return "Callback"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool grad_depends_output_data(int i, int o) const {
    if (!grad_depends_output_data_callback_)
      return true;
    return grad_depends_output_data_callback_(obj_, i, o);
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
    if (!grad_depends_input_data_callback_)
      return true;
    return grad_depends_input_data_callback_(obj_, i, j);
  }
};
}
#endif
