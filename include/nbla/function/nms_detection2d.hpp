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

/** NmsDetection2d
 */
#ifndef __NBLA_FUNCTION_NMS_DETECTION2D_HPP__
#define __NBLA_FUNCTION_NMS_DETECTION2D_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(NmsDetection2d, float, float, bool);

/** Non-Maximum Suppression (NMS) to 2D Object detector output.

    The full description of this function can be found at
    `functions.yaml` or a docstring of Python function
    `F.nms_detection2d`.

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param thresh Detection score threshold.
@param nms IoU threshold for Non-maximum suppression (NMS).
@param nms_per_class If true, NMS is applied for each class.

\ingroup FunctionImplGrp
 */
template <typename T>
class NmsDetection2d : public BaseFunction<float, float, bool> {
protected:
  float thresh_;
  float nms_;
  bool nms_per_class_;

public:
  NmsDetection2d(const Context &ctx, float thresh, float nms,
                 bool nms_per_class)
      : BaseFunction<float, float, bool>(ctx, thresh, nms, nms_per_class),
        thresh_(thresh), nms_(nms), nms_per_class_(nms_per_class) {}
  virtual ~NmsDetection2d() {}
  virtual shared_ptr<Function> copy() const {
    return create_NmsDetection2d(ctx_, thresh_, nms_, nms_per_class_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "NmsDetection2d"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API void forward_impl_per_class(const Variables &inputs,
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
