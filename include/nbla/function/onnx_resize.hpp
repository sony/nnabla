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

#ifndef NBLA_FUNCTION_ONNX_RESIZE_HPP
#define NBLA_FUNCTION_ONNX_RESIZE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(ONNXResize, const vector<float> &,
                              const vector<float> &, const vector<int> &,
                              const string &, const string &, float, int, float,
                              const string &);

enum class ResizeCoordTransformMode {
  HALF_PIXEL,
  PYTORCH_HALF_PIXEL,
  ALIGN_CORNERS,
  ASYMMETRIC,
  TF_HALF_PIXEL_FOR_NN,
  TF_CROP_AND_RESIZE,
};

enum class ResizeNearestMode {
  ROUND_PREFER_FLOOR,
  ROUND_PREFER_CEIL,
  FLOOR,
  CEIL,
};

struct ResizeOption {
  ResizeCoordTransformMode coord_mode = ResizeCoordTransformMode::HALF_PIXEL;
  float cubic_coeff_a = 0.f;
  bool exclude_outside = false;
  float extrapolation_value = 0.f;
  ResizeNearestMode nearest_mode = ResizeNearestMode::ROUND_PREFER_FLOOR;
  vector<float> roi;
  Size_t num_outer_dims = 0;
  Size_t num_resize_dims = 0;
  Size_t num_dims = 0;

public:
  ResizeOption() = default;

  ResizeOption(const ResizeCoordTransformMode coord_mode,
               const float cubic_coeff_a, const bool exclude_outside,
               const float extrapolation_value,
               const ResizeNearestMode nearest_mode, const vector<float> &roi,
               Size_t num_outer_dims, Size_t num_resize_dims)
      : coord_mode(coord_mode), cubic_coeff_a(cubic_coeff_a),
        exclude_outside(exclude_outside),
        extrapolation_value(extrapolation_value), nearest_mode(nearest_mode),
        roi(roi), num_outer_dims(num_outer_dims),
        num_resize_dims(num_resize_dims),
        num_dims(num_outer_dims + num_resize_dims) {}
};

/**
Resize an ND array with interpolation. This function provides a
compatible interface to ONNX Resize.

Inputs:
- x: N-D array.

Outputs:
- y: N-D array.

@tparam T Data type for computation.
@param roi RoIs for tf_crop_and_resize.
@param scales Scale factors along axes.
@param sizes Output size.
@param mode Interpolation mode chosen from ('nearest'|'linear'|'cubic').
@param coordinate_transformation_mode How to transform coordinates in the
resized tensor to coordinates in the original tensor.
@param cubic_coeff_a The coefficient for cubic interpolation.
@param exclude_outside Whether to set coefficients to zero when a sampling
location is outside the input tensor.
@param extrapolation_value An extrapolation value used when a sampling location
is outside the input tensor at tf_crop_and_resize mode.
@param nearest_mode Rounding mode for nearest-neighbor interpolation.

\ingroup FunctionImplGrp
 */
template <typename T>
class ONNXResize
    : public BaseFunction<const vector<float> &, const vector<float> &,
                          const vector<int> &, const string &, const string &,
                          float, int, float, const string &> {
protected:
  const vector<float> roi_;
  const vector<float> scales_;
  const vector<int> sizes_;
  const string mode_;
  const string coordinate_transformation_mode_;
  float cubic_coeff_a_;
  int exclude_outside_;
  float extrapolation_value_;
  const string nearest_mode_;

public:
  ONNXResize(const Context &ctx, const vector<float> &roi,
             const vector<float> &scales, const vector<int> &sizes,
             const string &mode, const string &coordinate_transformation_mode,
             float cubic_coeff_a, int exclude_outside,
             float extrapolation_value, const string &nearest_mode)
      : BaseFunction(ctx, roi, scales, sizes, mode,
                     coordinate_transformation_mode, cubic_coeff_a,
                     exclude_outside, extrapolation_value, nearest_mode),
        roi_(roi), scales_(scales), sizes_(sizes), mode_(mode),
        coordinate_transformation_mode_(coordinate_transformation_mode),
        cubic_coeff_a_(cubic_coeff_a), exclude_outside_(exclude_outside),
        extrapolation_value_(extrapolation_value), nearest_mode_(nearest_mode) {
  }
  virtual ~ONNXResize() {}
  virtual shared_ptr<Function> copy() const {
    return create_ONNXResize(
        ctx_, roi_, scales_, sizes_, mode_, coordinate_transformation_mode_,
        cubic_coeff_a_, exclude_outside_, extrapolation_value_, nearest_mode_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "ONNXResize"; }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  vector<float> actual_scales_;
  ResizeOption option_;

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
} // namespace nbla
#endif
