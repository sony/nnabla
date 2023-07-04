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

/** Shift
 */
#ifndef NBLA_FUNCTION_SHAPE_HPP
#define NBLA_FUNCTION_SHAPE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Shape, int, int);

/** Shape Get the shape of specified tensor.

Inputs:
- N-D array.

Outputs:
- 1-D array.

@tparam T Data type for computation.
@param start The start index of slice of shape array. If start axis is omit, the
slice
starts from 0.
@param end The end index of slice of shape array. If the end axis is omitted,
the axes
upto the last one will be included.
\ingroup FunctionImplGrp
*/
template <typename T> class Shape : public BaseFunction<int, int> {
protected:
  int start_;
  int end_;

public:
  Shape(const Context &ctx, int start, int end)
      : BaseFunction(ctx, start, end), start_(start), end_(end) {}
  virtual ~Shape() {}
  virtual shared_ptr<Function> copy() const {
    return create_Shape(ctx_, start_, end_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Shape"; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
} // namespace nbla
#endif
