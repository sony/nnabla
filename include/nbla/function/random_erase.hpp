// Copyright 2020,2021 Sony Corporation.
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

#ifndef NBLA_FUNCTION_RANDOM_ERASE_HPP
#define NBLA_FUNCTION_RANDOM_ERASE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <numeric>
#include <random>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(RandomErase, float, const vector<float> &,
                              const vector<float> &, const vector<float> &, int,
                              bool, bool, int, int, bool, bool);

/**
RandomErase randomly erase the inputs and replace with random values.

Inputs:
- N-D array of three or more dimensions

Outputs:
- N-D array.

@tparam T Data type for computation.
@param prob Probability to erase.
@param area_ratios Low and high of the area ratio range.
@param aspect_ratios Low and high of the aspect ratios range.
@param replacements Low and high of the replacement value range.
@param n Max number of patches to be erased.
@param share Use a same bounding box randomly picked over the feature dimension
when being True. Default is False.
@param inplace This option is obsolete and ignored. Output is never in-placed
with input.
@param base_axis
@param seed Random seed. When -1, seed is sampled from global random number
generator.
@param channel_last If True, the last dimension is considered as channel
dimension, a.k.a NHWC order.
@param ste_fine_grained Straight Through Estimator is fine-grained or not.
Default

\ingroup FunctionImplGrp
 */
template <typename T>
class RandomErase
    : public BaseFunction<float, const vector<float> &, const vector<float> &,
                          const vector<float> &, int, bool, bool, int, int,
                          bool, bool> {
protected:
  float prob_;
  const vector<float> area_ratios_;
  const vector<float> aspect_ratios_;
  const vector<float> replacements_;
  int n_;
  bool share_;
  int base_axis_;
  int seed_;
  bool channel_last_;
  bool ste_fine_grained_;

  bool save_rng_ = false;
  std::mt19937 rgen_, rgen_for_recompute_;

  NdArrayPtr random_coordinates_;

public:
  RandomErase(const Context &ctx, float prob, const vector<float> &area_ratios,
              const vector<float> &aspect_ratios,
              const vector<float> &replacements, int n, bool share,
              bool inplace, int base_axis, int seed, bool channel_last,
              bool ste_fine_grained)
      : BaseFunction(ctx, prob, area_ratios, aspect_ratios, replacements, n,
                     share, inplace, base_axis, seed, channel_last,
                     ste_fine_grained),
        prob_(prob), area_ratios_(area_ratios), aspect_ratios_(aspect_ratios),
        replacements_(replacements), n_(n), share_(share),
        base_axis_(base_axis), seed_(seed), channel_last_(channel_last),
        ste_fine_grained_(ste_fine_grained) {}
  virtual ~RandomErase() {}
  virtual shared_ptr<Function> copy() const {
    return create_RandomErase(ctx_, prob_, area_ratios_, aspect_ratios_,
                              replacements_, n_, share_,
                              false /* inplace is obsoleted. */, base_axis_,
                              seed_, channel_last_, ste_fine_grained_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "RandomErase"; }
  virtual bool need_setup_recompute(int o) const { return true; }
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
  NBLA_API virtual void setup_recompute_impl(const Variables &inputs,
                                             const Variables &outputs);
  NBLA_API virtual void recompute_impl(const Variables &inputs,
                                       const Variables &outputs);

  void random_erase(const Variables &inputs, const Variables &outputs,
                    std::mt19937 &rgen);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }
};
}
#endif
