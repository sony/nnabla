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

  // Instantiate functions used for the backward in test mode (batch_stat_ =
  // False)
  if (!batch_stat_) {
    identity_ = create_Identity(this->ctx_);
    add2_ = create_Add2(this->ctx_, false);
    sub2_ = create_Sub2(this->ctx_);
    mul2_ = create_Mul2(this->ctx_);
    add_scalar_ = create_AddScalar(this->ctx_, (T) this->eps_);
    pow_scalar_ = create_PowScalar(this->ctx_, (T)-0.5);
    std::vector<int> raxes;
    for (int i = 0; i < inputs[0]->ndim(); ++i) {
      if (i != axes_[0])
        raxes.push_back(i);
    }
    sum_ = create_Sum(this->ctx_, raxes, true);
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
    backward_impl_global(inputs, outputs, propagate_down, accum);
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

  // Common inputs wrt. gradient.
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

template <class T>
void BatchNormalization<T>::backward_impl_global(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2])) {
    return;
  }

  // Common inputs wrt. gradient.
  auto x = inputs[0];
  auto beta = inputs[1];
  auto gamma = inputs[2];
  auto rmean = inputs[3];
  auto rvar = inputs[4];
  auto y = outputs[0];

  // Running std
  shared_ptr<Variable> rstd_inv_sptr;
  shared_ptr<Variable> g_y_sptr;
  if (propagate_down[0] || propagate_down[2]) { // x or gamma
    // running std
    rstd_inv_sptr = make_shared<Variable>(rvar->shape());
    auto rstd_inv = rstd_inv_sptr.get();
    identity_->setup(Variables{rvar}, Variables{rstd_inv});
    identity_->forward(Variables{rvar}, Variables{rstd_inv});
    add_scalar_->setup(Variables{rstd_inv}, Variables{rstd_inv});
    add_scalar_->forward(Variables{rstd_inv}, Variables{rstd_inv});
    pow_scalar_->setup(Variables{rstd_inv}, Variables{rstd_inv});
    pow_scalar_->forward(Variables{rstd_inv}, Variables{rstd_inv});
    // g_y variable
    g_y_sptr = make_shared<Variable>(y->shape());
    g_y_sptr->set_data(y->grad());
  }

  // Gradient wrt. x.
  if (propagate_down[0]) {
    // gamma / rstd
    auto iv_sptr = make_shared<Variable>(beta->shape());
    auto rstd_inv = rstd_inv_sptr.get();
    auto iv = iv_sptr.get();
    mul2_->setup(Variables{gamma, rstd_inv}, Variables{iv});
    mul2_->forward(Variables{gamma, rstd_inv}, Variables{iv});
    // g_y * gamma / rstd
    auto g_x_tmp_sptr = make_shared<Variable>(x->shape());
    auto g_y = g_y_sptr.get();
    auto g_x_tmp = g_x_tmp_sptr.get();
    mul2_ = create_Mul2(this->ctx_);
    mul2_->setup(Variables{g_y, iv}, Variables{g_x_tmp});
    mul2_->forward(Variables{g_y, iv}, Variables{g_x_tmp});
    // accum
    auto g_x_sptr = make_shared<Variable>(x->shape());
    g_x_sptr->set_data(x->grad());
    auto g_x = g_x_sptr.get();
    if (accum[0]) {
      add2_->setup(Variables{g_x, g_x_tmp}, Variables{g_x});
      add2_->forward(Variables{g_x, g_x_tmp}, Variables{g_x});
    } else {
      identity_->setup(Variables{g_x_tmp}, Variables{g_x});
      identity_->forward(Variables{g_x_tmp}, Variables{g_x});
    }
  }

  // Gradient wrt. beta and gamma.
  if (propagate_down[1] || propagate_down[2]) { // beta and gamma
    NBLA_CHECK(propagate_down[1] && propagate_down[2], error_code::value,
               "'need_grad' of beta and gamma must be the same.");

    auto g_y = g_y_sptr.get();
    // 1. beta
    auto g_beta_tmp_sptr = make_shared<Variable>(beta->shape());
    auto g_beta_tmp = g_beta_tmp_sptr.get();
    sum_->setup(Variables{g_y}, Variables{g_beta_tmp});
    sum_->forward(Variables{g_y}, Variables{g_beta_tmp});
    auto g_beta_sptr = make_shared<Variable>(beta->shape());
    g_beta_sptr->set_data(beta->grad());
    auto g_beta = g_beta_sptr.get();

    if (accum[1]) {
      add2_->setup(Variables{g_beta, g_beta_tmp}, Variables{g_beta});
      add2_->forward(Variables{g_beta, g_beta_tmp}, Variables{g_beta});
    } else {
      identity_->setup(Variables{g_beta_tmp}, Variables{g_beta});
      identity_->forward(Variables{g_beta_tmp}, Variables{g_beta});
    }
    // 2. gamma
    // (x - rmean) / rstd
    auto iv_sptr = make_shared<Variable>(x->shape());
    auto iv = iv_sptr.get();
    sub2_->setup(Variables{x, rmean}, Variables{iv});
    sub2_->forward(Variables{x, rmean}, Variables{iv});
    auto rstd_inv = rstd_inv_sptr.get();
    mul2_->setup(Variables{iv, rstd_inv}, Variables{iv});
    mul2_->forward(Variables{iv, rstd_inv}, Variables{iv});
    // g_y * (x - rmean) / rstd
    mul2_ = create_Mul2(this->ctx_);
    mul2_->setup(Variables{g_y, iv}, Variables{iv});
    mul2_->forward(Variables{g_y, iv}, Variables{iv});
    // reduction
    auto g_gamma_tmp_sptr = make_shared<Variable>(gamma->shape());
    auto g_gamma_tmp = g_gamma_tmp_sptr.get();
    sum_->setup(Variables{iv}, Variables{g_gamma_tmp});
    sum_->forward(Variables{iv}, Variables{g_gamma_tmp});
    auto g_gamma_sptr = make_shared<Variable>(gamma->shape());
    g_gamma_sptr->set_data(gamma->grad());
    auto g_gamma = g_gamma_sptr.get();
    if (accum[2]) {
      add2_->setup(Variables{g_gamma, g_gamma_tmp}, Variables{g_gamma});
      add2_->forward(Variables{g_gamma, g_gamma_tmp}, Variables{g_gamma});
    } else {
      identity_->setup(Variables{g_gamma_tmp}, Variables{g_gamma});
      identity_->forward(Variables{g_gamma_tmp}, Variables{g_gamma});
    }
  }
}
}
