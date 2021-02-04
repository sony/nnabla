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

#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/add2.hpp>
#include <nbla/function/add_scalar.hpp>
#include <nbla/function/broadcast.hpp>
#include <nbla/function/div2.hpp>
#include <nbla/function/greater.hpp>
#include <nbla/function/greater_equal.hpp>
#include <nbla/function/identity.hpp>
#include <nbla/function/less.hpp>
#include <nbla/function/less_equal.hpp>
#include <nbla/function/max.hpp>
#include <nbla/function/maximum2.hpp>
#include <nbla/function/min.hpp>
#include <nbla/function/min_max_quantize.hpp>
#include <nbla/function/minimum2.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/mul_scalar.hpp>
#include <nbla/function/round.hpp>
#include <nbla/function/sub2.hpp>
#include <nbla/function/sum.hpp>

#include <nbla/variable.hpp>
#include <numeric>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(MinMaxQuantize, float, bool, bool, bool, float);

template <typename T>
void MinMaxQuantize<T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  // Check shape and dimensions
  auto x = inputs[0];
  auto qr_min = inputs[1];
  auto qr_max = inputs[2];
  auto ql_min = inputs[3];
  auto ql_max = inputs[4];

  // dimension check
  auto ndims = {ql_min->ndim(), ql_max->ndim(), qr_min->ndim(), qr_max->ndim()};

  NBLA_CHECK(std::equal(ndims.begin(), ndims.end() - 1, ndims.begin() + 1),
             error_code::value, "All input dimensions must be same.",
             "inputs[0] = %d, inputs[1] = %d, inputs[2] = %d, inputs[3] = %d, "
             "inputs[4] = %d.",
             x->ndim(), ql_min->ndim(), ql_max->ndim(), qr_min->ndim(),
             qr_max->ndim());
  // shape check
  auto shapes = {ql_min->shape(), ql_max->shape(), qr_min->shape(),
                 qr_max->shape()};
  NBLA_CHECK(std::equal(shapes.begin(), shapes.end() - 1, shapes.begin() + 1),
             error_code::value, "All shapes among ql_min, ql_max, qr_min, and "
                                "qr_max must be same.");
  // non-broadcast dimension check
  vector<int> none_axes;
  for (int i = 0; i < ql_min->ndim(); i++) {
    if (ql_min->shape()[i] != 1)
      none_axes.push_back(i);
  }
  NBLA_CHECK(std::all_of(none_axes.begin(), none_axes.end(),
                         [x, ql_min](int a) {
                           return x->shape()[a] == ql_min->shape()[a];
                         }),
             error_code::value, "x.shape and ql_min.shape must be same except "
                                "for i-th dimension where ql_min.shape[i] == "
                                "1.");

  // Copy and cast reduction axes for Min and Max
  vector<int> axes;
  for (int i = 0; i < ql_min->ndim(); i++) {
    if (ql_min->shape()[i] == 1)
      axes.push_back(i);
  }
  // Compute broadcast shape
  std::vector<int> bshape(x->shape().size());
  for (Shape_t::size_type i = 0; i < x->shape().size(); i++) {
    bshape[i] = x->shape()[i];
  }
  // Create functions needed to compute the min-max quantization
  identity_ = create_Identity(this->ctx_);
  round_ = create_Round(this->ctx_);
  add2_ = create_Add2(this->ctx_, false);
  sub2_ = create_Sub2(this->ctx_, false);
  mul2_ = create_Mul2(this->ctx_, false);
  div2_ = create_Div2(this->ctx_, false);
  minimum2_ = create_Minimum2(this->ctx_);
  maximum2_ = create_Maximum2(this->ctx_);
  mul_scalar_ = create_MulScalar(this->ctx_, (T)(decay_), false);
  mul_scalar2_ = create_MulScalar(this->ctx_, (T)(1.0 - decay_), false);
  min_ = create_Min(this->ctx_, axes, true, false, false);
  max_ = create_Max(this->ctx_, axes, true, false, false);
  broadcast_ = create_Broadcast(this->ctx_, bshape);
  if (ste_fine_grained_) {
    // pass gradient among min <= x <= max
    greater_equal_ = create_GreaterEqual(this->ctx_);
    less_equal_ = create_LessEqual(this->ctx_);
  }
  if (!x_min_max_ && !ema_) {
    greater_ = create_Greater(this->ctx_);
    less_ = create_Less(this->ctx_);
    sum_ = create_Sum(this->ctx_, axes, true);
  }

  // `setup` method is called in the forward passs for re-using the functions
  // and intermediate variable

  // Reshape output
  outputs[0]->reshape(x->shape(), true);
  scale_sptr_ = make_shared<Variable>(ql_min->shape());
}

template <typename T>
void MinMaxQuantize<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  // Variables
  auto x = inputs[0];
  auto qr_min = inputs[1];
  auto qr_max = inputs[2];
  auto ql_min = inputs[3];
  auto ql_max = inputs[4];
  auto scale = scale_sptr_.get();
  auto x_q = outputs[0];

  // Intermediate variable to be re-used
  auto iv_sptr = make_shared<Variable>(x->shape());
  auto iv = iv_sptr.get();

  /*
    Step-by-step computation:
    - if x_min_max is true,
      - x_min = min(x)
      - x_max = max(x)
    - if both x_min_max and ema is true,
      - ema(qr_min) = decay * qr_min_old + (1.0 - decay) * x_min
      - ema(qr_max) = decay * qr_max_old + (1.0 - decay) * x_max
    - scale = {qr_max - qr_min} / {ql_max - ql_min}
    - x_q = round( { min(max(x, qr_min), qr_max) - qr_min } / {s}) * s + qr_min
   */

  // min(x) and ema(qr_min) = decay * qr_min_old + (1.0 - decay) * x_min
  // working with shape of qr_min
  if (x_min_max_ && ema_) {
    execute(min_, Variables{x}, Variables{iv});
    execute(mul_scalar2_, Variables{iv}, Variables{iv});
    execute(mul_scalar_, Variables{qr_min}, Variables{qr_min});
    execute(add2_, Variables{qr_min, iv}, Variables{qr_min});
  } else if (x_min_max_ && !ema_) {
    execute(min_, Variables{x}, Variables{qr_min});
    execute(min_, Variables{x}, Variables{qr_min});
  }

  // max(x) and ema(qr_max) = decay * qr_max_old + (1.0 - decay) * x_max
  // working with shape of qr_min
  if (x_min_max_ && ema_) {
    execute(max_, Variables{x}, Variables{iv});
    execute(mul_scalar2_, Variables{iv}, Variables{iv});
    execute(mul_scalar_, Variables{qr_max}, Variables{qr_max});
    execute(add2_, Variables{qr_max, iv}, Variables{qr_max});
  } else if (x_min_max_ && !ema_) {
    execute(max_, Variables{x}, Variables{qr_max});
  }

  // TODO: optimize by writing down, naitve implementation is faster. Also we
  // have to write down in extensions.

  // ensure `qr_max - qr_min` of the scale should be greater than epsilon
  nudge_range(qr_min, qr_max);

  // scale = (qr_max - qr_min) / (ql_max - ql_min)
  // working with shape of qr_min
  execute(sub2_, Variables{qr_max, qr_min}, Variables{scale});
  execute(sub2_, Variables{ql_max, ql_min}, Variables{iv});
  execute(div2_, Variables{scale, iv}, Variables{scale});

  // nudge qr_min and qr_max
  auto qr_min_nudged_sptr = make_shared<Variable>(qr_min->shape());
  auto qr_min_nudged = qr_min_nudged_sptr.get();
  auto qr_max_nudged_sptr = make_shared<Variable>(qr_max->shape());
  auto qr_max_nudged = qr_max_nudged_sptr.get();
  nudge_qr_min_max(qr_min, qr_max, ql_min, ql_max, scale, qr_min_nudged,
                   qr_max_nudged);

  // broadcast
  auto _qr_min_sptr = make_shared<Variable>(x->shape());
  auto _qr_min = _qr_min_sptr.get();
  auto _qr_max_sptr = make_shared<Variable>(x->shape());
  auto _qr_max = _qr_max_sptr.get();
  auto _scale_sptr = make_shared<Variable>(x->shape());
  auto _scale = _scale_sptr.get();
  execute(broadcast_, Variables{qr_min_nudged}, Variables{_qr_min});
  execute(broadcast_, Variables{qr_max_nudged}, Variables{_qr_max});
  execute(broadcast_, Variables{scale}, Variables{_scale});

  // x_q = min(max(x, qr_min), qr_max)
  // working with shape of x
  execute(maximum2_, Variables{x, _qr_min}, Variables{x_q});
  execute(minimum2_, Variables{x_q, _qr_max}, Variables{x_q});

  // x_q = round( (x_q - qr_min) / scale ) * scale + qr_min
  // working with shape of x
  execute(sub2_, Variables{x_q, _qr_min}, Variables{x_q});
  execute(div2_, Variables{x_q, _scale}, Variables{x_q});
  execute(round_, Variables{x_q}, Variables{x_q});
  execute(mul2_, Variables{x_q, _scale}, Variables{x_q});
  execute(add2_, Variables{x_q, _qr_min}, Variables{x_q});
}

template <typename T>
void MinMaxQuantize<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  // The backward_impl is coupling with the forward_impl
  NBLA_CHECK(
      !(propagate_down[3] || propagate_down[4]), error_code::value,
      "Backward to x(inputs[0]), qr_min(inputs[1]), and/or qr_max(inputs[2])"
      "are only allowed.");
  auto x = inputs[0];
  auto qr_min = inputs[1];
  auto qr_max = inputs[2];
  auto ql_min = inputs[3];
  auto ql_max = inputs[4];
  auto scale = scale_sptr_.get();
  auto y = outputs[0];

  // scale is already nudged in forward_impl
  // nudge qr_min and qr_max
  auto qr_min_nudged_sptr = make_shared<Variable>(qr_min->shape());
  auto qr_min_nudged = qr_min_nudged_sptr.get();
  auto qr_max_nudged_sptr = make_shared<Variable>(qr_max->shape());
  auto qr_max_nudged = qr_max_nudged_sptr.get();
  nudge_qr_min_max(qr_min, qr_max, ql_min, ql_max, scale, qr_min_nudged,
                   qr_max_nudged);

  // broadcast first
  auto _qr_min_sptr = make_shared<Variable>(x->shape());
  auto _qr_min = _qr_min_sptr.get();
  auto _qr_max_sptr = make_shared<Variable>(x->shape());
  auto _qr_max = _qr_max_sptr.get();
  if (ste_fine_grained_ || !x_min_max_) {
    execute(broadcast_, Variables{qr_min_nudged}, Variables{_qr_min});
    execute(broadcast_, Variables{qr_max_nudged}, Variables{_qr_max});
  }

  // w.r.t. x
  if (propagate_down[0]) {
    auto g_x_sptr = make_shared<Variable>(x->shape());
    auto g_y_sptr = make_shared<Variable>(y->shape());
    g_x_sptr->set_data(x->grad());
    g_y_sptr->set_data(y->grad());
    auto g_x = g_x_sptr.get();
    auto g_y = g_y_sptr.get();
    if (ste_fine_grained_) {
      // mask
      auto iv0_sptr = make_shared<Variable>(x->shape());
      auto iv0 = iv0_sptr.get();
      auto iv1_sptr = make_shared<Variable>(x->shape());
      auto iv1 = iv1_sptr.get();
      execute(greater_equal_, Variables{x, _qr_min}, Variables{iv0});
      execute(less_equal_, Variables{x, _qr_max}, Variables{iv1});
      auto mask_sptr = make_shared<Variable>(x->shape());
      auto mask = mask_sptr.get();
      execute(mul2_, Variables{iv0, iv1}, Variables{mask});
      // mul mask
      execute(mul2_, Variables{mask, g_y}, Variables{mask});
      if (accum[0]) {
        execute(add2_, Variables{mask, g_x}, Variables{g_x});
      } else {
        execute(identity_, Variables{mask}, Variables{g_x});
      }
    } else {
      if (accum[0]) {
        execute(add2_, Variables{g_y, g_x}, Variables{g_x});
      } else {
        execute(identity_, Variables{g_y}, Variables{g_x});
      }
    }
  }

  if (x_min_max_ || ema_)
    return;

  auto iv_sptr0 = make_shared<Variable>(x->shape());
  auto iv0 = iv_sptr0.get();
  auto iv_sptr1 = make_shared<Variable>(qr_min->shape());
  auto iv1 = iv_sptr1.get();
  // w.r.t. qr_min
  if (propagate_down[1]) {
    // Compute sum(mask * g_y)
    // mask
    auto mask_sptr = make_shared<Variable>(x->shape());
    auto mask = mask_sptr.get();
    execute(less_, Variables{x, _qr_min}, Variables{mask});
    auto g_y_sptr = make_shared<Variable>(y->shape());
    auto g_y = g_y_sptr.get();
    g_y_sptr->set_data(y->grad());
    // mul mask
    execute(mul2_, Variables{mask, g_y}, Variables{iv0});
    // reduce
    execute(sum_, Variables{iv0}, Variables{iv1});
    auto g_qr_min_sptr = make_shared<Variable>(qr_min->shape());
    auto g_qr_min = g_qr_min_sptr.get();
    g_qr_min_sptr->set_data(qr_min->grad());
    if (accum[1]) {
      execute(add2_, Variables{g_qr_min, iv1}, Variables{g_qr_min});
    } else {
      execute(identity_, Variables{iv1}, Variables{g_qr_min});
    }
  }
  // w.r.t. qr_max
  if (propagate_down[2]) {
    // Compute sum(mask * g_y)
    // mask
    auto mask_sptr = make_shared<Variable>(x->shape());
    auto mask = mask_sptr.get();
    execute(greater_, Variables{x, _qr_max}, Variables{mask});
    auto g_y_sptr = make_shared<Variable>(y->shape());
    auto g_y = g_y_sptr.get();
    g_y_sptr->set_data(y->grad());
    // mul mask
    execute(mul2_, Variables{mask, g_y}, Variables{iv0});
    // reduce
    execute(sum_, Variables{iv0}, Variables{iv1});
    auto g_qr_max_sptr = make_shared<Variable>(qr_max->shape());
    auto g_qr_max = g_qr_max_sptr.get();
    g_qr_max_sptr->set_data(qr_max->grad());
    if (accum[2]) {
      execute(add2_, Variables{g_qr_max, iv1}, Variables{g_qr_max});
    } else {
      execute(identity_, Variables{iv1}, Variables{g_qr_max});
    }
  }
}

template <typename T>
void MinMaxQuantize<T>::nudge_range(Variable *qr_min, Variable *qr_max) {
  const T *qr_min_data = qr_min->get_data_pointer<T>(ctx_);
  T *qr_max_data = qr_max->cast_data_and_get_pointer<T>(ctx_);
  for (int i = 0; i < qr_min->size(); ++i) {
    if (qr_max_data[i] - qr_min_data[i] < this->eps_) {
      qr_max_data[i] = qr_min_data[i] + this->eps_;
    }
  }
}
template <typename T>
void MinMaxQuantize<T>::nudge_qr_min_max(Variable *qr_min, Variable *qr_max,
                                         Variable *ql_min, Variable *ql_max,
                                         Variable *scale,
                                         Variable *qr_min_nudged,
                                         Variable *qr_max_nudged) {
  auto qr_min_data = qr_min->get_data_pointer<T>(ctx_);
  auto ql_min_data = ql_min->get_data_pointer<T>(ctx_);
  auto ql_max_data = ql_max->get_data_pointer<T>(ctx_);
  auto scale_data = scale->get_data_pointer<T>(ctx_);
  auto qr_min_nudged_data = qr_min_nudged->cast_data_and_get_pointer<T>(ctx_);
  auto qr_max_nudged_data = qr_max_nudged->cast_data_and_get_pointer<T>(ctx_);

  T zero_point_nudged = T(0);
  for (int i = 0; i < qr_min->size(); ++i) {
    auto zero_point_from_min = ql_min_data[i] - qr_min_data[i] / scale_data[i];
    if (zero_point_from_min <= ql_min_data[i]) {
      zero_point_nudged = ql_min_data[i];
    } else if (zero_point_from_min >= ql_max_data[i]) {
      zero_point_nudged = ql_max_data[i];
    } else {
      zero_point_nudged = std::round(zero_point_from_min);
    }
    qr_min_nudged_data[i] =
        (ql_min_data[i] - zero_point_nudged) * scale_data[i];
    qr_max_nudged_data[i] =
        (ql_max_data[i] - zero_point_nudged) * scale_data[i];
  }
}
}
