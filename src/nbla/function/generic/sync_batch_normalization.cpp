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
#include <nbla/function/sync_batch_normalization.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(SyncBatchNormalization,
                              const std::shared_ptr<Communicator> &,
                              const std::string &, const vector<int> &, float,
                              float, bool);

template <typename T>
void SyncBatchNormalization<T>::setup_impl(const Variables &inputs,
                                           const Variables &outputs) {
  this->BatchNormalization<T>::setup_impl(inputs, outputs);

  // Get the number of the processes in the communicator group
  this->num_processes_ = this->comm_->find_group(this->group_).size();
}

template <class T>
void SyncBatchNormalization<T>::forward_impl_batch(const Variables &inputs,
                                                   const Variables &outputs) {
  // Check whether it outputs batch mean and var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
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
  T *v = batch_var->cast_data_and_get_pointer<T>(this->ctx_, true); // batch var
  // Inputs/Outputs
  T *rm = inputs[3]->cast_data_and_get_pointer<T>(this->ctx_); // running mean
  T *rv = inputs[4]->cast_data_and_get_pointer<T>(this->ctx_); // running var

  // Local-batch mean and sq-mean
  for (int i1 = 0; i1 < this->size1_; ++i1) {
    m[i1] = 0;
    v[i1] = 0;
    for (int i02 = 0; i02 < this->size02_; ++i02) {
      const int i0 = i02 / this->size2_;
      const int i2 = i02 % this->size2_;
      const int i = i0 * this->size12_ + i1 * this->size2_ + i2;
      const T value = x[i];
      m[i1] += value;
      v[i1] += value * value;
    }
    m[i1] /= this->size02_;
    v[i1] /= this->size02_;
  }

  // Sync between other processes
  this->comm_->all_reduce({batch_mean->data(), batch_var->data()}, false, false,
                          this->group_);

  /*
   * The all_reduce is conducted in GPU because the communicator for CPU
   * extension is not
   * implemented currently.
   * Synchronization between GPU array and CPU array is needed to get the
   * results of all_reduce in GPU.
   */
  m = batch_mean->cast_data_and_get_pointer<T>(this->ctx_);
  v = batch_var->cast_data_and_get_pointer<T>(this->ctx_);
  for (int i1 = 0; i1 < this->size1_; ++i1) {
    // Mean and variance calculation and their moving ones.
    m[i1] /= this->num_processes_;
    v[i1] = v[i1] / this->num_processes_ - m[i1] * m[i1];

    auto n = this->size02_ * this->num_processes_;
    // Moving mean and var
    rm[i1] = this->decay_rate_ * rm[i1] + (1 - this->decay_rate_) * m[i1];
    rv[i1] = this->decay_rate_ * rv[i1] +
             (1 - this->decay_rate_) * v[i1] * n / (n - 1);

    // v[i1] = 1 / std::sqrt(v[i1] + (T)eps_);
    // Subtract mean and divide by std, and apply beta and gamma.
    for (int i02 = 0; i02 < this->size02_; ++i02) {
      const int i0 = i02 / this->size2_;
      const int i2 = i02 % this->size2_;
      const int i = i0 * this->size12_ + i1 * this->size2_ + i2;
      const T stdvar = std::sqrt(v[i1] + (T) this->eps_);
      y[i] = (x[i] - m[i1]) * gamma[i1] / stdvar + beta[i1];
    }
  }
}

template <class T>
void SyncBatchNormalization<T>::backward_impl_batch(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2])) {
    return;
  }
  // Check whether it outputs batch mean/var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  if (outputs.size() == 3) {
    batch_mean = outputs[1];
    batch_var = outputs[2];
  }

  // Common inputs wrt. gradient.
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *m = batch_mean->get_data_pointer<T>(this->ctx_);
  const T *v = batch_var->get_data_pointer<T>(this->ctx_);
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);

  // Synchronize between other processes
  Variable buff_arr({this->size1_ * 2});
  T *buff = buff_arr.cast_data_and_get_pointer<T>(this->ctx_);
  T *sum_dy_ptr = buff;
  T *sum_dyx_ptr = buff + this->size1_;
  for (int i1 = 0; i1 < this->size1_; ++i1) {
    sum_dy_ptr[i1] = 0;
    sum_dyx_ptr[i1] = 0;
    for (int i02 = 0; i02 < this->size02_; ++i02) {
      const int i0 = i02 / this->size2_;
      const int i2 = i02 % this->size2_;
      const int i = i0 * this->size12_ + i1 * this->size2_ + i2;
      sum_dy_ptr[i1] += dy[i];
      sum_dyx_ptr[i1] += dy[i] * x[i];
    }
  }
  this->comm_->all_reduce(buff_arr.data(), false, false, this->group_);
  buff = buff_arr.cast_data_and_get_pointer<T>(this->ctx_);
  sum_dy_ptr = buff;
  sum_dyx_ptr = buff + this->size1_;

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

    // Compute gradient wrt mean and var respectively
    auto n = this->num_processes_ * this->size02_;
    // Compute gradient wrt x.
    for (int i1 = 0; i1 < this->size1_; ++i1) {
      // dm and dv are set if batch mean and var are used following functions
      // in computation graph.
      T dvar = g[i1] * sum_dyx_ptr[i1] - g[i1] * m[i1] * sum_dy_ptr[i1];
      dvar = dvar * (T)-0.5 * std::pow(v[i1] + (T) this->eps_, (T)-1.5) +
             (dv ? dv[i1] : (T)0);
      T dmean = g[i1] * sum_dy_ptr[i1];
      dmean = dmean * (-1 / std::sqrt(v[i1] + (T) this->eps_)) +
              (dm ? dm[i1] : (T)0);

      for (int i02 = 0; i02 < this->size02_; ++i02) {
        const int i0 = i02 / this->size2_;
        const int i2 = i02 % this->size2_;
        const int i = i0 * this->size12_ + i1 * this->size2_ + i2;
        const T grad = dy[i] * g[i1] / std::sqrt(v[i1] + (T) this->eps_) +
                       dvar * 2 * (x[i] - m[i1]) / n + dmean / n;
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
    for (int i1 = 0; i1 < this->size1_; ++i1) {
      T dbv = accum[1] ? db[i1] : (T)0;
      T dgv = accum[2] ? dg[i1] : (T)0;
      db[i1] = dbv + sum_dy_ptr[i1];
      dg[i1] = dgv + sum_dyx_ptr[i1] / std::sqrt(v[i1] + (T) this->eps_) -
               m[i1] / std::sqrt(v[i1] + (T) this->eps_) * sum_dy_ptr[i1];
    }
  }
}
}
