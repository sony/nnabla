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
#include <nbla/function/spectral_norm.hpp>
#include <nbla/variable.hpp>

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/functions.hpp>

#include <nbla/function/identity.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(SpectralNorm, int, int, float, bool);

namespace {
CgVariablePtr create_cgvariable_from_variable(Variable *var,
                                              const bool need_grad) {
  auto cg_var = make_shared<CgVariable>(var->shape(), need_grad);
  cg_var->variable()->set_data(var->data());
  cg_var->variable()->set_grad(var->grad());
  return cg_var;
}

CgVariablePtr norm_normalization(Context &ctx, CgVariablePtr x,
                                 const float eps) {
  // norm
  CgVariablePtr norm;
  norm = functions::pow_scalar(ctx, x, 2.0, false)[0];
  norm = functions::sum(ctx, norm, {0, 1}, true)[0];
  norm = functions::add_scalar(ctx, norm, eps, true)[0];
  norm = functions::pow_scalar(ctx, norm, 0.5, false)[0];
  // norm_normalization
  return functions::div2(ctx, x, norm, false)[0];
}

CgVariablePtr identity_with_inplace_output(Context &ctx, CgVariablePtr x,
                                           NdArrayPtr output) {
  return connect(make_shared<CgFunction>(create_Identity(ctx)), {x}, 1,
                 {output}, false)[0];
}
}

template <typename T>
CgVariablePtr SpectralNorm<T>::spectral_norm(const Variables &inputs,
                                             const Variables &outputs) {
  auto w = create_cgvariable_from_variable(inputs[0], true);
  auto u = create_cgvariable_from_variable(inputs[1], false);

  // pre transpose
  if (dim_ != 0) {
    vector<int> dims_transpose;
    dims_transpose.push_back(dim_);
    for (int i = 0; i < inputs[0]->ndim(); i++) {
      if (i != dim_) {
        dims_transpose.push_back(i);
      }
    }
    w = functions::transpose(ctx_, w, dims_transpose)[0];
  }

  vector<int> w_shape;
  for (auto s : w->variable()->shape()) {
    w_shape.push_back(s);
  }

  // reshape
  w = functions::reshape(ctx_, w, {d0_, d1_}, true)[0];
  u = functions::reshape(ctx_, u, {1, d0_}, true)[0];
  // `u_` is used for forward recalculation in backward.
  u_ = u->variable()->data();

  // power method
  CgVariablePtr v;
  for (int i = 0; i < itr_; i++) {
    // v
    v = functions::affine(ctx_, u, w, nullptr, 1)[0];
    v = norm_normalization(ctx_, v, eps_);
    v = functions::reshape(ctx_, v, {d1_, 1}, true)[0];
    // u
    u = functions::affine(ctx_, w, v, nullptr, 1)[0];
    u = norm_normalization(ctx_, u, eps_);
    u = functions::reshape(ctx_, u, {1, d0_}, true)[0];
  }

  // Iterate
  if (!test_) {
    u = identity_with_inplace_output(ctx_, u, inputs[1]->data());
  }
  u->set_persistent(true);

  u->set_need_grad(false);
  v->set_need_grad(false);

  // Spectral normalization
  auto wv = functions::affine(ctx_, w, v, nullptr, 1)[0];
  auto sigma = functions::affine(ctx_, u, wv, nullptr, 1)[0];
  auto w_sn = functions::div2(ctx_, w, sigma, false)[0];
  w_sn = functions::reshape(ctx_, w_sn, w_shape, true)[0];

  // post transpose
  if (dim_ != 0) {
    vector<int> dims_transpose;
    for (int i = 1; i < dim_ + 1; i++) {
      dims_transpose.push_back(i);
    }
    dims_transpose.push_back(0);
    for (int i = dim_ + 1; i < inputs[0]->ndim(); i++) {
      dims_transpose.push_back(i);
    }
    w_sn = functions::transpose(ctx_, w_sn, dims_transpose)[0];
  }

  return w_sn;
}

template <typename T>
CgVariablePtr
SpectralNorm<T>::spectral_norm_outer_most_dim(const Variables &inputs,
                                              const Variables &outputs) {
  auto w = create_cgvariable_from_variable(inputs[0], true);
  auto u = create_cgvariable_from_variable(inputs[1], false);

  vector<int> w_shape;
  for (auto s : inputs[0]->shape()) {
    w_shape.push_back(s);
  }

  // Reshape
  w = functions::reshape(ctx_, w, {d0_, d1_}, true)[0];
  u = functions::reshape(ctx_, u, {d1_, 1}, true)[0];
  // `u_` is used for forward recalculation in backward.
  u_ = u->variable()->data();

  // Power method
  CgVariablePtr v;
  for (int i = 0; i < itr_; i++) {
    // v
    v = functions::affine(ctx_, w, u, nullptr, 1)[0];
    v = norm_normalization(ctx_, v, eps_);
    v = functions::reshape(ctx_, v, {1, d0_}, true)[0];
    // u
    u = functions::affine(ctx_, v, w, nullptr, 1)[0];
    u = norm_normalization(ctx_, u, eps_);
    u = functions::reshape(ctx_, u, {d1_, 1}, true)[0];
  }

  // Iterate
  if (!test_) {
    u = identity_with_inplace_output(ctx_, u, inputs[1]->data());
  }
  u->set_persistent(true);

  u->set_need_grad(false);
  v->set_need_grad(false);

  // Spectral normalization
  auto vw = functions::affine(ctx_, v, w, nullptr, 1)[0];
  auto sigma = functions::affine(ctx_, vw, u, nullptr, 1)[0];
  auto w_sn = functions::div2(ctx_, w, sigma, false)[0];
  w_sn = functions::reshape(ctx_, w_sn, w_shape, true)[0];

  return w_sn;
}

template <typename T>
void SpectralNorm<T>::setup_impl(const Variables &inputs,
                                 const Variables &outputs) {
  const auto w = inputs[0];
  const auto ws = w->shape();

  NBLA_CHECK(0 <= dim_ && dim_ < w->ndim(), error_code::value,
             "`dim` must be `0 <= dim and dim < len(w.shape)`.");
  NBLA_CHECK(0 < itr_, error_code::value, "`itr` must be greater than 0.");
  NBLA_CHECK(0. < eps_, error_code::value, "`eps` must be greater than 0.");

  const bool is_outer_most_dim = dim_ == w->ndim() - 1;

  if (is_outer_most_dim) {
    d1_ = ws[w->ndim() - 1];
    d0_ = w->size() / d1_; // prod(w.shape[0:-1])
  } else {
    d0_ = ws[dim_];
    d1_ = w->size() / d0_; // prod(w.shape[1:])
  }

  // Naive graph composition
  CgVariablePtr last_out;
  if (is_outer_most_dim) {
    last_out = spectral_norm_outer_most_dim(inputs, outputs);
  } else {
    last_out = spectral_norm(inputs, outputs);
  }

  // `u_orig_` is used for forward recalculation in backprop
  u_orig_ = make_shared<NdArray>(inputs[1]->shape());

  // Replace the output variable in the last CgVariable with outputs[0].
  outputs[0]->reshape(last_out->variable()->shape(), true);
  last_out->variable()->set_data(outputs[0]->data());
  last_out->variable()->set_grad(outputs[0]->grad());

  // Call all setup again to ensure inplaced variables refer to the correct
  // array.
  std::unordered_set<CgFunctionPtr> fclosed;
  last_out->visit_function_recursive(last_out->parent(), fclosed,
                                     [](CgFunctionPtr fn) { fn->setup(); });
  this->last_output_cg_variable_ = last_out;
}

template <typename T>
void SpectralNorm<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {
  if (!test_) {
    // data region of `u` will be rewrited in this forward prop.
    // On the other hand, we need original `u` for forward recalculation in
    // backprop.
    // Therefore, we keep `u` into `u_org_` here.
    const Array *u_array = inputs[1]->data()->get(get_dtype<T>(), this->ctx_);
    Array *u_org_array = u_orig_->cast(get_dtype<T>(), this->ctx_, true);
    u_org_array->copy_from(u_array);
  }

  // Buffers are cleard during forward prop for memory optimization.
  // Forward calculation will be performed again during backward.
  last_output_cg_variable_->forward(true, true);
}

template <typename T>
void SpectralNorm<T>::backward_impl(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  // forward recalculation

  // Temporally restore `u` remembered in previous forward prop to prevent
  // double iteration of `u`
  u_->set_array(u_orig_->array());

  last_output_cg_variable_->forward(false, true);

  // Reset u_
  u_->set_array(inputs[1]->data()->array());

  // backward

  last_output_cg_variable_->backward(outputs[0]->grad(), true);
}
}
