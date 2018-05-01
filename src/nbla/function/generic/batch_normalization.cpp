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
#include <nbla/function/batch_normalization.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BatchNormalization, const vector<int> &, float,
                              float, bool);

template <typename T>
void BatchNormalization<T>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
  // Check axes
  NBLA_CHECK(axes_.size() == 1, error_code::not_implemented,
             "Specifying axis more than one is not supported so far.")

  // Check and parse shapes
  Shape_t shape_i = inputs[0]->shape();
  Size_t size = inputs[0]->size();
  Size_t size_axis = inputs[0]->size(axes_[0]);
  size0_ = size / size_axis;       // Batch size.
  size1_ = shape_i[axes_[0]];      // Size of specified axis.
  size2_ = size / size0_ / size1_; // Size of rest.
  size12_ = size1_ * size2_;
  size02_ = size0_ * size2_;
  NBLA_CHECK(size0_ * size1_ * size2_ == size, error_code::unclassified,
             "An error occurred during setup BatchNormalization function.");
  // Verify mean, var, beta and gamma dims.
  Shape_t shape_b = inputs[1]->shape();
  Shape_t shape_g = inputs[2]->shape();
  Shape_t shape_m = inputs[3]->shape();
  Shape_t shape_v = inputs[4]->shape();
  // Verify mean, var, beta and gamma shapes.
  Shape_t shape_check(shape_i.size(), 1);
  shape_check[axes_[0]] = shape_i[axes_[0]];
  NBLA_CHECK(shape_check == shape_b, error_code::value,
             "Shape of beta(inputs[1]) does not match. "
             "beta: (%s) != expected: (%s).",
             string_join(shape_b, string(", ")).c_str(),
             string_join(shape_check, string(", ")).c_str());
  NBLA_CHECK(shape_check == shape_g, error_code::value,
             "Shape of gamma(inputs[2]) does not match. "
             "gamma: (%s) != expected: (%s).",
             string_join(shape_g, string(", ")).c_str(),
             string_join(shape_check, string(", ")).c_str());
  NBLA_CHECK(shape_check == shape_m, error_code::value,
             "Shape of mean(inputs[3]) does not match. "
             "mean: (%s) != expected: (%s).",
             string_join(shape_m, string(", ")).c_str(),
             string_join(shape_check, string(", ")).c_str());
  NBLA_CHECK(shape_check == shape_v, error_code::value,
             "Shape of var(inputs[4]) does not match. "
             "var: (%s) != expected: (%s).",
             string_join(shape_v, string(", ")).c_str(),
             string_join(shape_check, string(", ")).c_str());

  // Check num of inputs and outputs.
  size_t noutputs = outputs.size();
  if (!batch_stat_) {
    NBLA_CHECK(
        noutputs == 1, error_code::value,
        "If batch_stat_ is false, it cannot output batch mean and variance.");
  }
  NBLA_CHECK(noutputs == 1 || noutputs == 3, error_code::value,
             "Number of outputs must be 1 or 3.");

  // Reshape outputs and/or temporary buffers.
  outputs[0]->reshape(shape_i, true);
  if (noutputs == 3) {
    outputs[1]->reshape(shape_b, true); // batch mean
    outputs[2]->reshape(shape_g, true); // batch var
  } else {
    mean_.reshape(shape_b, true); // batch mean
    var_.reshape(shape_g, true);  // batch var
  }
}

template <class T>
void BatchNormalization<T>::forward_impl(const Variables &inputs,
                                         const Variables &outputs) {
  if (batch_stat_) { // Training mode.
    forward_impl_batch(inputs, outputs);
  } else { // Testing mode.
    forward_impl_global(inputs, outputs);
  }
}

template <class T>
void BatchNormalization<T>::forward_impl_batch(const Variables &inputs,
                                               const Variables &outputs) {
  // Check whether it outputs batch mean and var.
  Variable *batch_mean = &mean_;
  Variable *batch_var = &var_;
  if (outputs.size() == 3) {
    batch_mean = outputs[1];
    batch_var = outputs[2];
  }
  // Inputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *beta = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *gamma = inputs[2]->get_data_pointer<T>(this->ctx_);
  // Output
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  T *m =
      batch_mean->cast_data_and_get_pointer<T>(this->ctx_, true); // batch mean
  T *v =
      batch_var->cast_data_and_get_pointer<T>(this->ctx_, true); // batch varf
  // Inputs/Outputs
  T *rm = inputs[3]->cast_data_and_get_pointer<T>(this->ctx_); // running mean
  T *rv = inputs[4]->cast_data_and_get_pointer<T>(this->ctx_); // running var

  // Main loop
  for (int i1 = 0; i1 < size1_; ++i1) {
    // Mean and variance calculation and their moving ones.
    // Batch mean and var
    m[i1] = 0;
    v[i1] = 0;
    for (int i02 = 0; i02 < size02_; ++i02) {
      const int i0 = i02 / size2_;
      const int i2 = i02 % size2_;
      const int i = i0 * size12_ + i1 * size2_ + i2;
      const T value = x[i];
      m[i1] += value;
      v[i1] += value * value;
    }
    m[i1] /= size02_;
    v[i1] = v[i1] / size02_ - m[i1] * m[i1];

    // Moving mean and var
    rm[i1] = decay_rate_ * rm[i1] + (1 - decay_rate_) * m[i1];
    rv[i1] = decay_rate_ * rv[i1] +
             (1 - decay_rate_) * v[i1] * size02_ / (size02_ - 1);

    // v[i1] = 1 / std::sqrt(v[i1] + (T)eps_);
    // Subtract mean and divide by std, and apply beta and gamma.
    for (int i02 = 0; i02 < size02_; ++i02) {
      const int i0 = i02 / size2_;
      const int i2 = i02 % size2_;
      const int i = i0 * size12_ + i1 * size2_ + i2;
      const T stdvar = std::sqrt(v[i1] + (T)eps_);
      y[i] = (x[i] - m[i1]) * gamma[i1] / stdvar + beta[i1];
    }
  }
}

template <class T>
void BatchNormalization<T>::forward_impl_global(const Variables &inputs,
                                                const Variables &outputs) {
  // Inputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *beta = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *gamma = inputs[2]->get_data_pointer<T>(this->ctx_);
  const T *rm = inputs[3]->get_data_pointer<T>(this->ctx_); // running mean
  const T *rv = inputs[4]->get_data_pointer<T>(this->ctx_); // running var
  // Output
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  // Subtract mean and divide by std, and apply beta and gamma.
  for (int i1 = 0; i1 < size1_; ++i1) {
    for (int i02 = 0; i02 < size02_; ++i02) {
      const int i0 = i02 / size2_;
      const int i2 = i02 % size2_;
      const int i = i0 * size12_ + i1 * size2_ + i2;
      const T mean = rm[i1];
      const T stdvar = std::sqrt(rv[i1] + (T)eps_);
      y[i] = (x[i] - mean) * gamma[i1] / stdvar + beta[i1];
    }
  }
}

template <class T>
void BatchNormalization<T>::backward_impl(const Variables &inputs,
                                          const Variables &outputs,
                                          const vector<bool> &propagate_down,
                                          const vector<bool> &accum) {
  if (batch_stat_) { // Training mode.
    backward_impl_batch(inputs, outputs, propagate_down, accum);
  } else { // Testing mode.
    NBLA_ERROR(error_code::not_implemented, "");
  }
}

template <class T>
void BatchNormalization<T>::backward_impl_batch(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2])) {
    return;
  }
  // Check whether it outputs batch mean/var.
  Variable *batch_mean = &mean_;
  Variable *batch_var = &var_;
  if (outputs.size() == 3) {
    batch_mean = outputs[1];
    batch_var = outputs[2];
  }

  // Commont inputs wrt. gradient.
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *m = batch_mean->get_data_pointer<T>(this->ctx_);
  const T *v = batch_var->get_data_pointer<T>(this->ctx_);
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);

  // Gradient wrt. x.
  if (propagate_down[0]) {
    T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
    const T *g = inputs[2]->get_data_pointer<T>(this->ctx_);
    const T *dm = nullptr;
    const T *dv = nullptr;
    if (outputs.size() == 3) {
      dm = batch_mean->get_grad_pointer<T>(this->ctx_);
      dv = batch_var->get_grad_pointer<T>(this->ctx_);
    }
    for (int i1 = 0; i1 < size1_; ++i1) {
      // Compute gradient wrt mean and var respectively
      T dvar = 0;
      T dmean = 0;
      T tmp = 0;
      for (int i02 = 0; i02 < size02_; ++i02) {
        const int i0 = i02 / size2_;
        const int i2 = i02 % size2_;
        const int i = i0 * size12_ + i1 * size2_ + i2;
        const T dxh = dy[i] * g[i1]; // Grad of x hat.
        const T cx = x[i] - m[i1];   // x - mean
        dvar += dxh * cx;
        dmean += dxh;
        tmp += cx;
      }
      // dm and dv are set if batch mean and var are used following functions
      // in computation graph.
      dvar = dvar * (T)-0.5 * std::pow(v[i1] + (T)eps_, (T)-1.5) +
             (dv ? dv[i1] : (T)0);
      dmean = dmean * (-1 / std::sqrt(v[i1] + (T)eps_)) +
              dvar * (-2) * tmp / (size02_) + (dm ? dm[i1] : (T)0);
      // Compute gradient wrt x.
      for (int i02 = 0; i02 < size02_; ++i02) {
        const int i0 = i02 / size2_;
        const int i2 = i02 % size2_;
        const int i = i0 * size12_ + i1 * size2_ + i2;
        const T grad = dy[i] * g[i1] / std::sqrt(v[i1] + (T)eps_) +
                       dvar * 2 * (x[i] - m[i1]) / (size02_) +
                       dmean / (size02_);
        if (accum[0])
          dx[i] += grad;
        else
          dx[i] = grad;
      }
    }
  }

  if (propagate_down[1] || propagate_down[2]) { // beta and gamma
    NBLA_CHECK(propagate_down[1] && propagate_down[2], error_code::value,
               "'need_grad' of beta and gamma must be the same.");
    T *db = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[1]);
    T *dg = inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[2]);
    for (int i1 = 0; i1 < size1_; ++i1) {
      T dbv = accum[1] ? db[i1] : (T)0;
      T dgv = accum[2] ? dg[i1] : (T)0;
      for (int i02 = 0; i02 < size02_; ++i02) {
        const int i0 = i02 / size2_;
        const int i2 = i02 % size2_;
        const int i = i0 * size12_ + i1 * size2_ + i2;
        dbv += dy[i];
        dgv += dy[i] * (x[i] - m[i1]) / std::sqrt(v[i1] + (T)eps_);
      }
      db[i1] = dbv;
      dg[i1] = dgv;
    }
  }
}
}
