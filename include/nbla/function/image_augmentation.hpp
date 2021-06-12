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

/** ImageAugmentation
 */
#ifndef __NBLA_FUNCTION_IMAGEAUGMENTATION_HPP__
#define __NBLA_FUNCTION_IMAGEAUGMENTATION_HPP__

#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <random>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(ImageAugmentation, const vector<int> &,
                              const vector<int> &, float, float, float, float,
                              float, bool, bool, float, bool, float, float,
                              bool, float, int);

/** ImageAugmentation randomly alters the input image.

Inputs:
- N-D array of three or more dimensions

Outputs:
- N-D array.

@tparam T Data type for computation and score variable.
@param shape The output image data size.
@param pad Border padding values for each spatial axis (height and width).
Padding will be added both sides of the dimension.
@param min_scale The minimum scale ratio when randomly scaling the image. For
example, to scale down to 0.8 times the size of the original image, specify
"0.8". To not apply random scaling, set both min_scale and max_scale to "1.0".
@param max_scale The maximum scale ratio when randomly scaling the image. For
example, to scale down to 2 times the size of the original image, specify "2.0".
@param angle The rotation angle range in radians when randomly rotating the
image. The image is randomly rotated in the -Angle to +Angle range. For example,
to rotate in a +-15 degree range, specify "0.26" (15 degrees/360 degrees * 2PI).
To not apply random rotation, specify "0.0".
@param aspect_ratio The aspect ratio range when randomly deforming the image.
For example, to deform aspect ratio of image from 1:1.3 to 1.3:1, specify "1.3".
To not apply random deforming, specify "1.0".
@param distortion The distortion range when randomly distorting the image. To
not apply distortion, specify "0.0".
@param flip_lr Whether to randomly flip the image horizontally at 50%
probability.
@param flip_ud Whether to randomly flip the image vertically at 50% probability.
@param brightness The range of values to randomly add to the brightness. A
random value in the -Brightness to +Brightness range is added to the brightness.
For example, to vary the brightness in the -0.05 to +0.05 range, specify "0.05".
To not apply random addition to brightness, specify "0.0".
@param brightness_each Whether to apply the random addition to brightness (as
specified by brightness) to each color channel. True: brightness is added based
on a different random number for each channel. False: brightness is added based
on a random number common to all channels.
@param contrast The range in which to randomly very the image contrast. The
contrast is varied in the 1/Contrast times to Contrast times range. The output
brightness is equal to (input - contrast_center) * contrast + contrast_center.
For example, to vary the contrast in the 0.91 times to 1.1 times range,
specify "1.1". To not apply random contrast variation, specify "1.0".
@param contrast_center Intensity center used for applying contrast.
@param contrast_each Whether to apply the random contrast variation (as
specified by contrast) to each color channel. True: contrast is varied based on
a different random number for each channel. False: contrast is varied based on a
random number common to all channels.
@param noise Sigma of normal random number to be added.
@param seed Random seed.
\ingroup FunctionImplGrp
*/
template <typename T>
class ImageAugmentation
    : public BaseFunction<const vector<int> &, const vector<int> &, float,
                          float, float, float, float, bool, bool, float, bool,
                          float, float, bool, float, int> {
protected:
  const vector<int> shape_;
  const vector<int> pad_;
  float min_scale_;
  float max_scale_;
  float angle_;
  float aspect_ratio_;
  float distortion_;
  bool flip_lr_;
  bool flip_ud_;
  float brightness_;
  bool brightness_each_;
  float contrast_;
  float contrast_center_;
  bool contrast_each_;
  float noise_;
  int seed_;

  bool save_rng_ = false;
  std::mt19937 rgen_, rgen_for_recompute_;
  std::bernoulli_distribution rdist_;

public:
  ImageAugmentation(const Context &ctx, const vector<int> &shape,
                    const vector<int> &pad, float min_scale, float max_scale,
                    float angle, float aspect_ratio, float distortion,
                    bool flip_lr, bool flip_ud, float brightness,
                    bool brightness_each, float contrast, float contrast_center,
                    bool contrast_each, float noise, int seed)
      : BaseFunction(ctx, shape, pad, min_scale, max_scale, angle, aspect_ratio,
                     distortion, flip_lr, flip_ud, brightness, brightness_each,
                     contrast, contrast_center, contrast_each, noise, seed),
        shape_(shape), pad_(pad), min_scale_(min_scale), max_scale_(max_scale),
        angle_(angle), aspect_ratio_(aspect_ratio), distortion_(distortion),
        flip_lr_(flip_lr), flip_ud_(flip_ud), brightness_(brightness),
        brightness_each_(brightness_each), contrast_(contrast),
        contrast_center_(contrast_center), contrast_each_(contrast_each),
        noise_(noise), seed_(seed) {}
  virtual ~ImageAugmentation() {}
  virtual shared_ptr<Function> copy() const {
    return create_ImageAugmentation(
        ctx_, shape_, pad_, min_scale_, max_scale_, angle_, aspect_ratio_,
        distortion_, flip_lr_, flip_ud_, brightness_, brightness_each_,
        contrast_, contrast_center_, contrast_each_, noise_, seed_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "ImageAugmentation"; }
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
  void image_augmentation(const Variables &inputs, const Variables &outputs,
                          std::mt19937 &rgen);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }
};
}
#endif
