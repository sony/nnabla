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

#ifndef NBLA_FUNCTION_ONNX_NON_MAX_SUPPRESSION_HPP
#define NBLA_FUNCTION_ONNX_NON_MAX_SUPPRESSION_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(ONNXNonMaxSuppression, int, int, float, float);

/**
Non-Maximum Suppression (NMS) to 2D Object detector output.

    The full description of this function can be found at
    `functions.yaml` or a docstring of Python function
    `F.onnx_non_max_suppression`.

Inputs:
- 3-D bounding boxes array.
- 3-D scores array.

Outputs:
- 2-D selected indices array.

@tparam T Data type for computation.
@param max_output_boxes_per_class The maximum number of boxes selected
per batch per class.
@param iou_threshold IoU threshold for Non-maximum suppression (NMS).
@param score_threshold Detection score threshold.

\ingroup FunctionImplGrp
 */
template <typename T>
class ONNXNonMaxSuppression : public BaseFunction<int, int, float, float> {
protected:
  int center_point_box_;
  int max_output_boxes_per_class_;
  float iou_threshold_;
  float score_threshold_;

  Size_t batch_size_;
  Size_t num_boxes_;
  Size_t num_classes_;

public:
  ONNXNonMaxSuppression(const Context &ctx, int center_point_box,
                        int max_output_boxes_per_class, float iou_threshold,
                        float score_threshold)
      : BaseFunction(ctx, center_point_box, max_output_boxes_per_class,
                     iou_threshold, score_threshold),
        center_point_box_(center_point_box),
        max_output_boxes_per_class_(max_output_boxes_per_class),
        iou_threshold_(iou_threshold), score_threshold_(score_threshold) {}
  virtual ~ONNXNonMaxSuppression() {}
  virtual shared_ptr<Function> copy() const {
    return create_ONNXNonMaxSuppression(ctx_, center_point_box_,
                                        max_output_boxes_per_class_,
                                        iou_threshold_, score_threshold_);
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
  virtual string name() { return "ONNXNonMaxSuppression"; }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  virtual void non_max_suppression(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }
};
} // namespace nbla
#endif
