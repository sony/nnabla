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

/** Unpooling
*/
#ifndef __NBLA_FUNCTION_UNPOOLING_HPP__
#define __NBLA_FUNCTION_UNPOOLING_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function_registry.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Unpooling, const vector<int> &, bool);

/** Unpooling function which upsample a feature map with an integer factor for
each dimension.

The vector size of the kernel parameter corresponds to the number of dimensions
of the unpooling operation. e.g) If (N, M, L, H, W)-shaped tensor is given as an
input and kernel is (K, K), unpooling operation is applied to the last two
dimensions, then the resulting output size will be (N, M, L, KH, KW).

Inputs:
- N-d array.

Outputs:
- N-d array.

@tparam T Data type for computation.
@param kernel A list of upscaling factor of image.
\ingroup FunctionImplGrp
*/

template <typename T>
class Unpooling : public BaseFunction<const vector<int> &, bool> {
protected:
  vector<int> kernel_;
  bool channel_last_;

public:
  Unpooling(const Context &ctx, const vector<int> &kernel, bool channel_last)
      : BaseFunction(ctx, kernel, channel_last), kernel_(kernel),
        channel_last_(channel_last) {}

  virtual ~Unpooling() {}
  virtual shared_ptr<Function> copy() const {
    return create_Unpooling(this->ctx_, this->kernel_, this->channel_last_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Unpooling"; }
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
    return false;
  }

private:
  void unpooling_forward_recursive(const Variable *inp, Variable *outp,
                                   const T *x, T *y, int x_offset, int y_offset,
                                   int dim);
  void unpooling_backward_recursive(Variable *outp, const Variable *inp, T *dx,
                                    const T *dy, int x_offset, int y_offset,
                                    int dim);
};
}
#endif
