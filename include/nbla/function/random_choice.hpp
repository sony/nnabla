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

#ifndef NBLA_FUNCTION_RANDOM_CHOICE_HPP
#define NBLA_FUNCTION_RANDOM_CHOICE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>
#include <random>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(RandomChoice, const vector<int> &, bool, int);

/** Generate random samples from population `x` with selection
    probabilities determined by the relative weights `w`. The number
    of samples to draw is given by the product of 'shape' dimensions
    and the samples are returned with the given `shape`. By default
    samples are drawn with replacement, i.e. selection of a specific
    population member is solely determined by its associated
    weight. Sampling without replacement, where any population member
    may be drawn only once, is used if `replace` is set to False.

Inputs:

- x: N-D array from which a random sample is generated.
- w: N-D array of associated weights of elements in `x`.

Outputs:

- N-D array

@tparam T Data type for computation.

@param shape Number and shape of generated samples.

@param replace Whether sampling is with or without replacement.

@param seed: Random seed.

\ingroup FunctionImplGrp
 */
template <typename T>
class RandomChoice : public BaseFunction<const vector<int> &, bool, int> {
protected:
  const vector<int> shape_;
  bool replace_;
  int seed_;
  std::mt19937 rgen_;
  Variable idxbuf_;   // stores chosen indices for backward
  Size_t outer_loop_; // product of batch dimensions
  Size_t inner_loop_; // product of shape dimensions

public:
  RandomChoice(const Context &ctx, const vector<int> &shape, bool replace,
               int seed)
      : BaseFunction(ctx, shape, replace, seed), shape_(shape),
        replace_(replace), seed_(seed) {}
  virtual ~RandomChoice() {}
  virtual shared_ptr<Function> copy() const {
    return create_RandomChoice(ctx_, shape_, replace_, seed_);
  }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "RandomChoice"; }

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
