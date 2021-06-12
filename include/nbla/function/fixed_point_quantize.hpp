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

// fixed_point_quantize.hpp
#ifndef __NBLA_FUNCTION_FIXED_POINT_QUANTIZE_HPP__
#define __NBLA_FUNCTION_FIXED_POINT_QUANTIZE_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(FixedPointQuantize, bool, int, float, bool);

/** FixedPointQuantize quantizes values in fixed-point number representation.

@tparam T Data type for computation.
@param sign Indicate the signed number or the unsigned number. Default is true.
@param n Bit width used. Note that `sign` consumes one bit. \f$n-1\f$ is used
for number representation in `signed` case.
@param delta Step size.
@param quantize If true, quantize input, otherwise not.
@param ste_fine_grained If true, STE is not 1.

 */
template <typename T>
class FixedPointQuantize : public BaseFunction<bool, int, float, bool> {
protected:
  const bool sign_;   // Indicate the signed fixed-point number or the unsigned
                      // fixed-point number. The default is true, use the signed
                      // fixed-point number.
  const int n_;       // Bit width used, take care that `sign` consumes one-bit.
                      // \f$n-1\f$ is used for number representation in `signed`
                      // case.
  const float delta_; // Step size
  const bool ste_fine_grained_;

  float max_; // upper bound in in fixed-point number region.
  float min_; // lower bound in in fixed-point number region.

public:
  FixedPointQuantize(const Context &ctx, bool sign, int n, float delta,
                     bool ste_fine_grained)
      : BaseFunction(ctx, sign, n, delta, ste_fine_grained), sign_(sign), n_(n),
        delta_(delta), ste_fine_grained_(ste_fine_grained) {}
  virtual ~FixedPointQuantize() {}
  virtual shared_ptr<Function> copy() const {
    return create_FixedPointQuantize(ctx_, sign_, n_, delta_,
                                     ste_fine_grained_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "FixedPointQuantize"; }
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
