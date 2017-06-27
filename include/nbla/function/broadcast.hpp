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

/** Broadcast
 */
#ifndef __NBLA_FUNCTION_BROADCAST_HPP__
#define __NBLA_FUNCTION_BROADCAST_HPP__

#ifndef NBLA_BROADCAST_MAX_DIM
/// Maximum number of dimensions Broadcast function supports.
#define NBLA_BROADCAST_MAX_DIM 8
#endif

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Broadcast, const vector<int> &);

/** Broadcast N-D array to shape.

@param shape The shape input array broadcasted to. Dimensions broadcasted in
    input array must be size one.
 */
template <typename T>
class Broadcast : public BaseFunction<const vector<int> &> {
protected:
  const vector<int> shape_;

  Variable stride_x_;
  Variable shape_y_;

public:
  Broadcast(const Context &ctx, const vector<int> &shape)
      : BaseFunction<const vector<int> &>(ctx, shape), shape_(shape) {}
  virtual ~Broadcast() {}
  virtual shared_ptr<Function> copy() const {
    return create_Broadcast(ctx_, shape_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Broadcast"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
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
};
}
#endif
