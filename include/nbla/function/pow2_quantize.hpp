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

/** Pow2Quantize
 */
#ifndef __NBLA_FUNCTION_POW2QUANTIZE_HPP__
#define __NBLA_FUNCTION_POW2QUANTIZE_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Pow2Quantize, bool, bool, int, int, bool);

/**
This function quantizes values in the power of 2 number representation,
in other words, it is linear (uniform) quantization in :math:`log_2` domain.

@tparam T Data type for computation.
@param sign Indicate the signed number or the unsigned number. Default is true.
@param with_zero Indicate using zero as a quantized value. Default is true. Note
that `zero` consumes one bit.
@param n Bit width used. Note that `sign` consumes one bit. \f$n-1\f$ is used
for number representation in `signed` case.
@param m \f$2^m\f$ is the upper bound of the dynamic range and \f$-2^m\f$ is the
lower bound, \f$m \in \mathcal{Z}\f$.
@param quantize If true, quantize input, otherwise not.
@param ste_fine_grained If true, STE is not 1.

 */
template <typename T>
class Pow2Quantize : public BaseFunction<bool, bool, int, int, bool> {
protected:
  const bool sign_; // Indicate the signed fixed-point number or the unsigned
                    // fixed-point number. The default is true, use the signed
                    // fixed-point number.
  const bool
      with_zero_; // Indicate using zero as a quantized value. Default is true.

  const int n_; // Bit width used, take care that `sign` consumes one-bit.
                // :math:`n-1` is used for number representation in `signed`
                // case.
  const int m_; // \f$2^m\f$ is upper bound and \f$-2^m\f$ is lower bound.
  const bool ste_fine_grained_;

  float p_max_; // upper bound in positive region
  float p_min_; // lower bound in positive region
  float pruning_threshold_;

public:
  Pow2Quantize(const Context &ctx, bool sign, bool with_zero, int n, int m,
               bool ste_fine_grained)
      : BaseFunction<bool, bool, int, int, bool>(ctx, sign, with_zero, n, m,
                                                 ste_fine_grained),
        sign_(sign), with_zero_(with_zero), n_(n), m_(m),
        ste_fine_grained_(ste_fine_grained) {}
  virtual ~Pow2Quantize() {}
  virtual shared_ptr<Function> copy() const {
    return create_Pow2Quantize(ctx_, sign_, with_zero_, n_, m_,
                               ste_fine_grained_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Pow2Quantize"; }
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
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (ste_fine_grained_) {
      return true;
    }
    return false;
  }
};
}
#endif
