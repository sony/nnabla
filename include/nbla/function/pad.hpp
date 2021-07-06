// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_PAD_HPP
#define NBLA_FUNCTION_PAD_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Pad, const vector<int> &, const string &, float);

/** Pad the input N-D array `x` over the number dimensions given by
half the length of the `pad_width` iterable, where every two values in
`pad_width` determine the before and after pad size of an axis
starting from the last (innermost) dimension and continuing towards
the first (outermost) dimension. The `pad_width` iterable must hold an
even number of positive values which may cover all or fewer dimensions
of the input variable `x`.

Padding is performed according to the requested `mode`. If pad mode is
"constant" then all padded elements are set to `constant_value`. If
pad mode is "reflect" then the padded alements are populated with the
reflection of the vector mirrored on the first and last values of the
vector along each axis.

Inputs:
- x: N-D array variable.
- pad_width: Vector with an even number of padding values.
- mode: Padding mode string, either 'constant' or 'reflect'.
- constant_value - Fill value if mode is 'constant'.

Outputs:
- Padded N-D array with the same number of dimensions as the input.

@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */

typedef struct { int first, second; } PadItem;
typedef std::vector<PadItem> PadList;

template <typename T>
class Pad : public BaseFunction<const vector<int> &, const string &, float> {
protected:
  const vector<int> pad_width_;
  const string mode_string_;
  const T constant_value_;

  enum { PAD_CONSTANT, PAD_REFLECT, PAD_REPEAT } pad_mode_;
  PadList padding_;
  Shape_t x_stride_;
  Shape_t y_stride_;
  Shape_t y_shape_;

public:
  Pad(const Context &ctx, const vector<int> &pad_width, const string &mode,
      float constant_value)
      : BaseFunction(ctx, pad_width, mode, constant_value),
        pad_width_(pad_width), mode_string_(mode),
        constant_value_(constant_value) {}
  virtual ~Pad() {}
  virtual shared_ptr<Function> copy() const {
    return create_Pad(ctx_, pad_width_, mode_string_, constant_value_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Pad"; }
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
};
}
#endif
