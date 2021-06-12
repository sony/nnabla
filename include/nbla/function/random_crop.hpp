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

/** RandomCrop
 */
#ifndef __NBLA_FUNCTION_RANDOMCROP_HPP__
#define __NBLA_FUNCTION_RANDOMCROP_HPP__

#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <random>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(RandomCrop, const vector<int> &, int, int);

/** RandomCrop randomly extracts a portion of an array.

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param shape The data size to extract. For example, to randomly extract a
portion of the image (3,48,48) from a 3,64,64 image, specify (3,48,48).
\ingroup FunctionImplGrp
*/
template <typename T>
class RandomCrop : public BaseFunction<const vector<int> &, int, int> {
protected:
  const vector<int> shape_;
  int base_axis_;

  int size_;
  int dim_offset_;

  // These settings are array to realize different slice amount for each data.
  vector<vector<int>> start_;
  vector<vector<int>> stop_;
  vector<vector<int>> step_;

  int seed_;
  bool save_rng_ = false;
  std::mt19937 rgen_, rgen_for_recompute_;

public:
  RandomCrop(const Context &ctx, const vector<int> &shape, int base_axis,
             int seed)
      : BaseFunction(ctx, shape, base_axis, seed), shape_(shape),
        base_axis_(base_axis), size_(1), dim_offset_(0), seed_(seed) {}
  virtual ~RandomCrop() {}
  virtual shared_ptr<Function> copy() const {
    return create_RandomCrop(ctx_, shape_, base_axis_, seed_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "RandomCrop"; }
  virtual vector<string> allowed_array_classes() {
    return vector<string>{"CpuArray"};
  }
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
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }

private:
  void slice_forward_recursive(const Variable *inp, Variable *outp, const T *x,
                               T *y, int x_offset, int y_offset, int dim,
                               int &slice_index);
  void slice_backward_recursive(Variable *outp, const Variable *inp, T *dx,
                                const T *dy, int x_offset, int y_offset,
                                int dim, int &slice_index);

  void random_crop(const Variables &inputs, const Variables &outputs,
                   std::mt19937 &rgen);
};
}
#endif
