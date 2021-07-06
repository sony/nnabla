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

#ifndef NBLA_FUNCTION_RNN_HPP
#define NBLA_FUNCTION_RNN_HPP

#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>
#include <nbla/variable.hpp>

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/utils.hpp>
#include <nbla/computation_graph/variable.hpp>

#include <nbla/function/affine.hpp>
#include <nbla/function/concatenate.hpp>
#include <nbla/function/relu.hpp>
#include <nbla/function/reshape.hpp>
#include <nbla/function/sink.hpp>
#include <nbla/function/split.hpp>
#include <nbla/function/stack.hpp>
#include <nbla/function/tanh.hpp>
#include <nbla/function/transpose.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(RNN, int, const string &, float, bool, bool);

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T>
class RNN : public BaseFunction<int, const string &, float, bool, bool> {
protected:
  int num_layers_;
  const string nonlinearity_;
  float dropout_;
  bool bidirectional_;
  bool training_;

  int seq_len_;
  int input_dim_;
  int hidden_size_;
  int num_directions_;
  int batch_size_;
  bool weight_exists_;
  bool bias_exists_;
  vector<CgVariablePtr> ys_, hn_;
  shared_ptr<CgVariable> x_, h_, w0_, w_, b_;

public:
  RNN(const Context &ctx, int num_layers, const string &nonlinearity,
      float dropout, bool bidirectional, bool training)
      : BaseFunction(ctx, num_layers, nonlinearity, dropout, bidirectional,
                     training),
        num_layers_(num_layers), nonlinearity_(nonlinearity), dropout_(dropout),
        bidirectional_(bidirectional), training_(training) {}
  virtual ~RNN() {}
  virtual shared_ptr<Function> copy() const {
    return create_RNN(ctx_, num_layers_, nonlinearity_, dropout_,
                      bidirectional_, training_);
  }
  virtual int min_inputs() { return 3; }
  virtual int min_outputs() { return 2; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>(),
                          get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "RNN"; }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void forward_impl_training(const Variables &inputs,
                                              const Variables &outputs);
  NBLA_API virtual void forward_impl_inference(const Variables &inputs,
                                               const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }

private:
  vector<vector<CgVariablePtr>> create_fixed_length_rnn_graph(
      shared_ptr<CgVariable> in_x, shared_ptr<CgVariable> in_h,
      shared_ptr<CgVariable> in_w0, shared_ptr<CgVariable> in_w,
      shared_ptr<CgVariable> in_b);
  shared_ptr<CgVariable> rnn_cell(shared_ptr<CgVariable> x,
                                  shared_ptr<CgVariable> h,
                                  shared_ptr<CgVariable> w,
                                  shared_ptr<CgVariable> b);
};
}
#endif
