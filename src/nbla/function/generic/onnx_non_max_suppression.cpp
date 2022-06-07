// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#include <array>
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/onnx_non_max_suppression.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
static T calculate_overlap_with_range(T xa_begin, T xa_end, T xb_begin,
                                      T xb_end) {
  return std::min(xa_end, xb_end) - std::max(xa_begin, xb_begin);
}

template <typename T>
static T calculate_overlap_with_center(T xa_center, T wa, T xb_center, T wb) {
  return calculate_overlap_with_range(xa_center - wa / 2, xa_center + wa / 2,
                                      xb_center - wb / 2, xb_center + wb / 2);
}

template <typename T>
static T calculate_iou(const T *a, const T *b, int center_point_box) {
  // intersection
  T wa, ha, wb, hb, wi, hi;
  if (center_point_box == 0) {
    // box format: [y1, x1, y2, x2]
    // (x1, y1) and (x2, y2) are the coordinates of any diagonal pair of
    // box corners. This means (x1, y1) is not always the top left corner
    // of a bounding box.
    T ya_bgn, ya_end, xa_bgn, xa_end;
    T yb_bgn, yb_end, xb_bgn, xb_end;
    std::tie(ya_bgn, ya_end) = std::minmax(a[0], a[2]);
    std::tie(xa_bgn, xa_end) = std::minmax(a[1], a[3]);
    std::tie(yb_bgn, yb_end) = std::minmax(b[0], b[2]);
    std::tie(xb_bgn, xb_end) = std::minmax(b[1], b[3]);
    wa = xa_end - xa_bgn;
    ha = ya_end - ya_bgn;
    wb = xb_end - xb_bgn;
    hb = yb_end - yb_bgn;
    wi = calculate_overlap_with_range(xa_bgn, xa_end, xb_bgn, xb_end);
    hi = calculate_overlap_with_range(ya_bgn, ya_end, yb_bgn, yb_end);
  } else if (center_point_box == 1) {
    // box format: [x_center, y_center, width, height]
    const T xa_center = a[0];
    const T ya_center = a[1];
    const T xb_center = b[0];
    const T yb_center = b[1];
    wa = a[2];
    ha = a[3];
    wb = b[2];
    hb = b[3];
    wi = calculate_overlap_with_center(xa_center, wa, xb_center, wb);
    hi = calculate_overlap_with_center(ya_center, ha, yb_center, hb);
  } else {
    NBLA_ERROR(error_code::value, "center_point_box must be 0 or 1.");
  }

  if (wi <= 0 || hi <= 0)
    return 0;

  T intersection = wi * hi;

  // Union
  T union_ = wa * ha + wb * hb - intersection;

  // IoU
  return intersection / union_;
}

template <typename T>
static void non_max_suppression_impl(
    const T *boxes,  // (batch_size, num_boxes, 4)
    const T *scores, // (batch_size, num_classes, num_boxes)
    Size_t batch_size, Size_t num_boxes, Size_t num_classes,
    int center_point_box, size_t max_output_boxes, float iou_threshold,
    float score_threshold,
    vector<std::array<Size_t, 3>> &selected_indices // (num_selected_indices, 3)
    ) {
  // This implementation is based on NmsDetection2d<T>::forward_impl_per_class.
  // Differences from NmsDetection2d:
  // - The input/output format
  // - Support for center_point_box and max_output_boxes_per_class

  if (max_output_boxes == 0) {
    return;
  }

  vector<Size_t> indexes(num_boxes);
  vector<bool> selected_boxes(num_boxes);

  // NMS per class
  for (Size_t b = 0; b < batch_size; b++) {
    for (Size_t k = 0; k < num_classes; k++) {
      // Initialize buffer for sort
      for (Size_t i = 0; i < num_boxes; ++i) {
        indexes[i] = i;
      }
      // Sort indexes
      const T *scores_per_class = scores + (b * num_classes + k) * num_boxes;
      std::sort(indexes.begin(), indexes.end(),
                [&scores_per_class](Size_t i, Size_t j) {
                  return scores_per_class[i] > scores_per_class[j];
                });

      // Initialize selected or suppressed flags
      for (Size_t i = 0; i < num_boxes; ++i) {
        // Suppress by score
        selected_boxes[i] = (scores_per_class[i] >= score_threshold);
      }

      // NMS
      for (Size_t i = 0; i < num_boxes; ++i) {
        const Size_t box_index = indexes[i];
        if (!selected_boxes[box_index]) {
          continue;
        }
        const T *box = boxes + (b * num_boxes + box_index) * 4;
        for (Size_t j = i + 1; j < num_boxes; ++j) {
          const Size_t box_index2 = indexes[j];
          if (!selected_boxes[box_index2]) {
            continue;
          }
          // Suppress by IoU
          const T *box2 = boxes + (b * num_boxes + box_index2) * 4;
          const T iou = calculate_iou(box, box2, center_point_box);
          if (iou > iou_threshold) {
            selected_boxes[box_index2] = false;
          }
        }
      }

      // Fill outputs
      size_t count = 0;
      for (Size_t i = 0; i < num_boxes; ++i) {
        const Size_t box_index = indexes[i];
        if (selected_boxes[box_index]) {
          selected_indices.push_back({b, k, box_index});
          count += 1;
          if (count >= max_output_boxes) {
            break;
          }
        }
      }
    }
  }
}

NBLA_REGISTER_FUNCTION_SOURCE(ONNXNonMaxSuppression, int, int, float, float);

template <typename T>
void ONNXNonMaxSuppression<T>::setup_impl(const Variables &inputs,
                                          const Variables &outputs) {
  // Check the shape of boxes
  const Shape_t boxes_shape = inputs[0]->shape();
  NBLA_CHECK(boxes_shape.size() == 3, error_code::value,
             "The number of dimension of boxes must be 3. Given %d.",
             boxes_shape.size());
  NBLA_CHECK(boxes_shape[2] == 4, error_code::value,
             "The shape of boxes is illegal: The 2nd element (starting from 0) "
             "of the shape must be 4. Given %lld.",
             static_cast<long long>(boxes_shape[2]));

  batch_size_ = inputs[0]->shape()[0];
  num_boxes_ = inputs[0]->shape()[1];

  // Check the shape of scores
  const Shape_t scores_shape = inputs[1]->shape();
  NBLA_CHECK(scores_shape.size() == 3, error_code::value,
             "The number of dimension of scores must be 3. Given %d.",
             scores_shape.size());
  NBLA_CHECK(scores_shape[0] == batch_size_, error_code::value,
             "The shape of boxes is illegal: The 0th element "
             "of the shape must be same as batch size (%lld). Given %lld.",
             static_cast<long long>(batch_size_),
             static_cast<long long>(scores_shape[0]));
  NBLA_CHECK(scores_shape[2] == num_boxes_, error_code::value,
             "The shape of boxes is illegal: The 2nd element of the shape must "
             "be same as the number of boxes (%lld). Given %lld.",
             static_cast<long long>(num_boxes_),
             static_cast<long long>(scores_shape[2]));

  num_classes_ = inputs[1]->shape()[1];

  // Check arguments
  NBLA_CHECK(0 <= center_point_box_ && center_point_box_ <= 1,
             error_code::value, "center_point_box must be 0 or 1. Given %d.",
             center_point_box_);
  NBLA_CHECK(max_output_boxes_per_class_ >= 0, error_code::value,
             "max_output_boxes_per_class must be non-negative. Given %d.",
             max_output_boxes_per_class_);

  // Peform forward computation to determine the output shape.
  non_max_suppression(inputs, outputs);
}

template <typename T>
void ONNXNonMaxSuppression<T>::forward_impl(const Variables &inputs,
                                            const Variables &outputs) {
  // Forward is done at setup_impl() because the output shape is calculated
  // during forward computation.
}

template <typename T>
void ONNXNonMaxSuppression<T>::non_max_suppression(const Variables &inputs,
                                                   const Variables &outputs) {
  // Inputs
  const T *boxes = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *scores = inputs[1]->get_data_pointer<T>(this->ctx_);

  vector<std::array<Size_t, 3>> selected_indices;
  selected_indices.reserve(batch_size_ * num_classes_ *
                           max_output_boxes_per_class_);
  non_max_suppression_impl(boxes, scores, batch_size_, num_boxes_, num_classes_,
                           center_point_box_, max_output_boxes_per_class_,
                           iou_threshold_, score_threshold_, selected_indices);

  // Setup the output shape
  const Size_t num_selected_boxes = selected_indices.size();
  const Size_t num_indices = 3; // [batch_index, class_index, box_index]
  outputs[0]->reshape({num_selected_boxes, num_indices}, true);

  // Outputs
  size_t *y = outputs[0]->cast_data_and_get_pointer<size_t>(this->ctx_, true);
  for (Size_t i = 0; i < num_selected_boxes; ++i) {
    const auto &indices = selected_indices[i];
    y[i * num_indices + 0] = static_cast<size_t>(indices[0]);
    y[i * num_indices + 1] = static_cast<size_t>(indices[1]);
    y[i * num_indices + 2] = static_cast<size_t>(indices[2]);
  }
}

template <typename T>
void ONNXNonMaxSuppression<T>::backward_impl(const Variables &inputs,
                                             const Variables &outputs,
                                             const vector<bool> &propagate_down,
                                             const vector<bool> &accum) {
  // not supported.
}
}
