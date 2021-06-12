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

/** INQAffine
 */
#ifndef __NBLA_FUNCTION_INQAFFINE_HPP__
#define __NBLA_FUNCTION_INQAFFINE_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function/affine.hpp>
#include <nbla/function_registry.hpp>

#include <random>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(INQAffine, int, int, const vector<int> &,
                              const string &, int);

/** This function implements an INQ affine layer. During training, the
weights are sequentially quantized to power-of-two values, which allows
the training of a multiplierless network.

Using `inq_iterations`, one can specify after how many forward passes
half of the learnable weights are fixed and quantized to powers-of-two.
After reaching the last value in `inq_iterations`, all weights are fixed.

Please note that the weights are quantized in the forward pass. Therefore,
in order to make sure that we only have power-of-two values, one needs to do
a final call to `forward` as the weights might have been updated by the
solver.

For more details, please refer to the reference.

Reference:
Zhou A, Yao A, Guo Y, Xu L, Chen Y. Incremental network quantization:
Towards lossless CNNs with low-precision weights.
<https://arxiv.org/abs/1702.03044>

Inputs (\f$B\f$ is base_axis):
- Input N-D array with shape
  (\f$M_0 \times ... \times M_{B-1} \times D_B \times ... \times D_N\f$).
  Dimensions before and after base_axis are flattened as if it is a matrix.
- Weight matrix with shape (\f$(D_B \times ... \times D_N) \times L\f$)
- Indicator matrix with shape (\f$(D_B \times ... \times D_N) \times
L\f$) where `0` indicates learnable weights and `1` indicates fixed weights
- (optional) Bias vector (\f$L\f$)

Outputs:
- \f$(B + 1)\f$-D array. (\f$ M_0 \times ... \times M_{B-1} \times L \f$)

@tparam T Data type for computation.
@param base_axis Base axis of BinaryConnectAffine operation. Dimensions up to
base_axis is treated as sample dimension.
@param num_bits Number of bits per weight. Needs to be >= 2.
@param inq_iterations Vector of integer values which give after how many
forward passes we fix 50% of the learnable weights.
@param selection_algorithm Chooses algorithm for selection of weights that
we want to fix ("largest_abs" ... fix weights with largest absolute value,
"random" ... fix all learnable weights randomly with a probability of 50%)
@param seed Random seed
 */
template <typename T, typename T1>
class INQAffine
    : public BaseFunction<int, int, const vector<int> &, const string &, int> {
protected:
  int base_axis_;
  int num_bits_;
  const vector<int> inq_iterations_;
  const string selection_algorithm_;
  int seed_;

  Variable old_weights_;
  Variable old_indicators_;
  int minibatch_counter_;
  shared_ptr<Function> affine_;
  std::mt19937 rgen_;
  std::bernoulli_distribution rdist_;

public:
  INQAffine(const Context &ctx, int base_axis, int num_bits,
            const vector<int> &inq_iterations,
            const string &selection_algorithm, int seed)
      : BaseFunction<int, int, const vector<int> &, const string &, int>(
            ctx, base_axis, num_bits, inq_iterations, selection_algorithm,
            seed),
        base_axis_(base_axis), num_bits_(num_bits),
        inq_iterations_(inq_iterations),
        selection_algorithm_(selection_algorithm), seed_(seed) {}
  virtual ~INQAffine() {}
  virtual shared_ptr<Function> copy() const {
    return create_INQAffine(ctx_, base_axis_, num_bits_, inq_iterations_,
                            selection_algorithm_, seed_);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T1>(),
                          get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 3; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "INQAffine"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
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
  NBLA_API virtual void recompute_impl(const Variables &inputs,
                                       const Variables &outputs);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (i == 0 && j == 1) {
      return true;
    }
    if (i == 1 && j == 0) {
      return true;
    }
    return false;
  }
  virtual bool overwrite_input_data_in_forward_impl(int i) const {
    if (i == 1 || i == 2) {
      return true;
    }
    return false;
  }
};
}
#endif
