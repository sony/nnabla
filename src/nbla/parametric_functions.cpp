// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Include nnabla header files

#include <assert.h>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>

#include <nbla/exception.hpp>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/context.hpp>
#include <nbla/function.hpp>
#include <nbla/function/affine.hpp>
#include <nbla/function/batch_normalization.hpp>
#include <nbla/function/convolution.hpp>
#include <nbla/function/deconvolution.hpp>
#include <nbla/global_context.hpp>
#include <nbla/parametric_functions.hpp>
using std::make_shared;

namespace nbla {

CgVariablePtr make_parameter(Shape_t shape, Initializer *initializer,
                             bool need_grad = true) {
  auto parameter = make_shared<CgVariable>(shape, need_grad);
  initializer->initialize(parameter->variable()->data());
  return parameter;
}

ParameterDirectory::ParameterDirectory(
    string scope_path, shared_ptr<dict_type> param_dict,
    shared_ptr<ordered_keys_type> ordered_keys)
    : scope_path_(scope_path), param_dict_(param_dict),
      ordered_keys_(ordered_keys) {}

ParameterDirectory::ParameterDirectory()
    : scope_path_(""),
      param_dict_(make_shared<unordered_map<string, CgVariablePtr>>()),
      ordered_keys_(make_shared<vector<string>>()) {}

ParameterDirectory ParameterDirectory::operator[](string name) {
  auto new_scope_path = scope_path_ + '/' + name;
  return ParameterDirectory(new_scope_path, param_dict_, ordered_keys_);
}

vector<pair<string, VariablePtr>> ParameterDirectory::get_parameters() {
  // Return parameters in order of ordered_keys. The items are filtered out
  // according to the scope_path_
  vector<pair<string, VariablePtr>> parameters;
  for (auto k = ordered_keys_->begin(); k != ordered_keys_->end(); k++) {
    if (scope_path_.size() == 0) {
      auto p = param_dict_->find(*k);
      parameters.push_back({p->first, p->second->variable()});
    } else if ((k->size() >= scope_path_.size()) &&
               std::equal(std::begin(scope_path_), std::end(scope_path_),
                          std::begin(*k))) {
      auto p = param_dict_->find(*k);
      parameters.push_back({p->first, p->second->variable()});
    }
  }
  return parameters;
}

CgVariablePtr ParameterDirectory::get_parameter(string name) {
  // Parameter name.
  auto param_path = scope_path_ + "/" + name;

  // Search exist one.
  auto it = param_dict_->find(param_path);
  if (it != param_dict_->end()) {
    return it->second;
  }

  return nullptr;
}

CgVariablePtr
ParameterDirectory::get_parameter_or_create(string name, Shape_t shape,
                                            Initializer *initializer,
                                            bool need_grad = true) {
  // Parameter name.
  auto param_path = scope_path_ + "/" + name;

  // Search exist one.
  auto it = param_dict_->find(param_path);
  if (it != param_dict_->end()) {
    NBLA_CHECK(shape == it->second->variable()->shape(),
               error_code::value,
               "Parameter \"%s\" already exists but the shape you passed is mismatch."
               "the shape of existed paremeter: (%s) != the shape you passed: (%s).",
               param_path.c_str(),
               string_join(it->second->variable()->shape(), ", ").c_str(),
               string_join(shape, ", ").c_str());

    return it->second;
  }

  // Create a new one and initialize with the initializer.
  auto parameter = make_parameter(shape, initializer, need_grad);
  param_dict_->insert({param_path, parameter});
  ordered_keys_->push_back(param_path);

  return parameter;
}

ParameterDirectory ParameterDirectory::create_deep_copy() {
  auto param_dict = make_shared<dict_type>();
  auto ordered_keys = make_shared<ordered_keys_type>(*ordered_keys_.get());

  // get ctx
  auto ctx = SingletonManager::get<GlobalContext>()->get_current_context();

  for (auto p = param_dict_->begin(); p != param_dict_->end(); p++) {
    param_dict->insert({p->first, p->second->create_deep_copy(ctx, false)});
  }

  return ParameterDirectory("", param_dict, ordered_keys);
}

namespace parametric_functions {

AffineOpts::AffineOpts()
    : with_bias_(true), fix_parameters_(false), w_init_(nullptr),
      b_init_(nullptr) {}

AffineOpts &AffineOpts::with_bias(bool with_bias) {
  with_bias_ = with_bias;
  return *this;
}

AffineOpts &AffineOpts::fix_parameters(bool val) {
  fix_parameters_ = val;
  return *this;
}

AffineOpts &AffineOpts::w_init(Initializer *w_init) {
  w_init_ = w_init;
  return *this;
}

AffineOpts &AffineOpts::b_init(Initializer *b_init) {
  b_init_ = b_init;
  return *this;
}

bool AffineOpts::with_bias() { return with_bias_; }

bool AffineOpts::fix_parameters() { return fix_parameters_; }

Initializer *AffineOpts::w_init() { return w_init_; }

Initializer *AffineOpts::b_init() { return b_init_; }

ConvolutionOpts::ConvolutionOpts()
    : with_bias_(true), fix_parameters_(false), w_init_(nullptr),
      b_init_(nullptr) {}

ConvolutionOpts &ConvolutionOpts::group(int val) {
  base_.group(val);
  return *this;
}

ConvolutionOpts &ConvolutionOpts::pad(const vector<int> &val) {
  base_.pad(val);
  return *this;
}

ConvolutionOpts &ConvolutionOpts::stride(const vector<int> &val) {
  base_.stride(val);
  return *this;
}

ConvolutionOpts &ConvolutionOpts::dilation(const vector<int> &val) {
  base_.dilation(val);
  return *this;
}

int ConvolutionOpts::group() { return base_.group(); }

const vector<int> &ConvolutionOpts::pad() const { return base_.pad(); }

const vector<int> &ConvolutionOpts::stride() const { return base_.stride(); }

const vector<int> &ConvolutionOpts::dilation() const {
  return base_.dilation();
}

ConvolutionOpts &ConvolutionOpts::with_bias(bool with_bias) {
  with_bias_ = with_bias;
  return *this;
}

ConvolutionOpts &ConvolutionOpts::fix_parameters(bool val) {
  fix_parameters_ = val;
  return *this;
}

ConvolutionOpts &ConvolutionOpts::w_init(Initializer *w_init) {
  w_init_ = w_init;
  return *this;
}

ConvolutionOpts &ConvolutionOpts::b_init(Initializer *b_init) {
  b_init_ = b_init;
  return *this;
}

bool ConvolutionOpts::with_bias() { return with_bias_; }

bool ConvolutionOpts::fix_parameters() { return fix_parameters_; }

Initializer *ConvolutionOpts::w_init() { return w_init_; }

Initializer *ConvolutionOpts::b_init() { return b_init_; }

BatchNormalizationOpts::BatchNormalizationOpts() : fix_parameters_(false) {}

BatchNormalizationOpts &BatchNormalizationOpts::axes(const vector<int> &val) {
  base_.axes(val);
  return *this;
}

BatchNormalizationOpts &BatchNormalizationOpts::decay_rate(float val) {
  base_.decay_rate(val);
  return *this;
}

BatchNormalizationOpts &BatchNormalizationOpts::eps(float val) {
  base_.eps(val);
  return *this;
}

const vector<int> &BatchNormalizationOpts::axes() const { return base_.axes(); }

float BatchNormalizationOpts::decay_rate() { return base_.decay_rate(); }

float BatchNormalizationOpts::eps() { return base_.eps(); }

BatchNormalizationOpts &BatchNormalizationOpts::fix_parameters(bool val) {
  fix_parameters_ = val;
  return *this;
}

bool BatchNormalizationOpts::fix_parameters() { return fix_parameters_; }

vector<CgVariablePtr> affine(Context &ctx, CgVariablePtr x, int base_axis,
                             int n_out, ParameterDirectory parameters,
                             bool with_bias, bool fix_parameters,
                             Initializer *w_init, Initializer *b_init) {

  shared_ptr<Initializer> shared_w_init;
  shared_ptr<Initializer> shared_b_init;

  Shape_t shape_x = x->variable()->shape();
  long int n_in = 1;
  for (int i = base_axis; i < shape_x.size(); i++)
    n_in *= shape_x[i];

  if (w_init == nullptr) {
    float parameter_range = calc_uniform_lim_glorot(n_in, n_out, 1);
    shared_w_init =
        make_shared<UniformInitializer>(-parameter_range, parameter_range);
    w_init = shared_w_init.get();
  }

  CgVariablePtr affine_w =
      parameters.get_parameter_or_create("affine/W", {n_in, n_out}, w_init);

  if (with_bias) {

    if (b_init == nullptr) {
      shared_b_init = make_shared<ConstantInitializer>();
      b_init = shared_b_init.get();
    }

    CgVariablePtr affine_b =
        parameters.get_parameter_or_create("affine/b", {n_out}, b_init);

    bool execute_forward = true;
    return connect(make_shared<CgFunction>(create_Affine(ctx, base_axis)),
                   {x, affine_w, affine_b}, execute_forward);

  } else {

    bool execute_forward = true;
    return connect(make_shared<CgFunction>(create_Affine(ctx, base_axis)),
                   {x, affine_w}, execute_forward);
  }
}

CgVariablePtr affine(CgVariablePtr x, int base_axis, int n_out,
                     ParameterDirectory parameters, AffineOpts affine_opts) {
  auto global_ctx =
      SingletonManager::get<GlobalContext>()->get_current_context();
  return affine(global_ctx, x, base_axis, n_out, parameters,
                affine_opts.with_bias(), affine_opts.fix_parameters(),
                affine_opts.w_init(), affine_opts.b_init())[0];
}

vector<CgVariablePtr>
convolution(Context &ctx, CgVariablePtr x, int base_axis, int n_map_out,
            const vector<int> &kernel, const vector<int> &pad,
            const vector<int> &stride, const vector<int> &dilation, int group,
            ParameterDirectory parameters, bool with_bias, bool fix_parameters,
            Initializer *w_init, Initializer *b_init) {

  shared_ptr<Initializer> shared_w_init;
  shared_ptr<Initializer> shared_b_init;

  Shape_t shape_x = x->variable()->shape();
  long int n_map_in = shape_x[base_axis];
  Shape_t shape_w = {n_map_out, int(n_map_in / group)};

  int kernel_dim_product = 1;
  for (int kernel_k : kernel) {
    shape_w.push_back(kernel_k);
    kernel_dim_product *= kernel_k;
  }

  if (w_init == nullptr) {
    float parameter_range =
        calc_uniform_lim_glorot(n_map_in, n_map_out, kernel_dim_product);
    shared_w_init =
        make_shared<UniformInitializer>(-parameter_range, parameter_range);
    w_init = shared_w_init.get();
  }

  CgVariablePtr conv_w =
      parameters.get_parameter_or_create("conv/W", shape_w, w_init);

  if (with_bias) {

    if (b_init == nullptr) {
      shared_b_init = make_shared<ConstantInitializer>();
      b_init = shared_b_init.get();
    }

    CgVariablePtr conv_b =
        parameters.get_parameter_or_create("conv/b", {n_map_out}, b_init);

    bool execute_forward = true;
    return connect(make_shared<CgFunction>(create_Convolution(
                       ctx, base_axis, pad, stride, dilation, group)),
                   {x, conv_w, conv_b}, execute_forward);

  } else {

    bool execute_forward = true;
    return connect(make_shared<CgFunction>(create_Convolution(
                       ctx, base_axis, pad, stride, dilation, group)),
                   {x, conv_w}, execute_forward);
  }
}

CgVariablePtr convolution(CgVariablePtr x, int base_axis, int n_map_out,
                          const vector<int> &kernel,
                          ParameterDirectory parameters,
                          ConvolutionOpts conv_opts) {
  auto global_ctx =
      SingletonManager::get<GlobalContext>()->get_current_context();
  return convolution(global_ctx, x, base_axis, n_map_out, kernel,
                     conv_opts.pad(), conv_opts.stride(), conv_opts.dilation(),
                     conv_opts.group(), parameters, conv_opts.with_bias(),
                     conv_opts.fix_parameters(), conv_opts.w_init(),
                     conv_opts.b_init())[0];
}

vector<CgVariablePtr>
deconvolution(Context &ctx, CgVariablePtr x, int base_axis, int n_map_out,
              const vector<int> &kernel, const vector<int> &pad,
              const vector<int> &stride, const vector<int> &dilation, int group,
              ParameterDirectory parameters, bool with_bias,
              bool fix_parameters, Initializer *w_init, Initializer *b_init) {

  shared_ptr<UniformInitializer> shared_w_init;
  shared_ptr<ConstantInitializer> shared_b_init;

  Shape_t shape_x = x->variable()->shape();
  long int n_map_in = shape_x[base_axis];

  Shape_t shape_w = {n_map_in, int(n_map_out / group)};

  int kernel_dim_product = 1;
  for (int kernel_k : kernel) {
    shape_w.push_back(kernel_k);
    kernel_dim_product *= kernel_k;
  }

  if (w_init == nullptr) {
    float parameter_range =
        calc_uniform_lim_glorot(n_map_in, n_map_out, kernel_dim_product);
    shared_w_init =
        make_shared<UniformInitializer>(-parameter_range, parameter_range);
    w_init = shared_w_init.get();
  }

  CgVariablePtr deconv_w =
      parameters.get_parameter_or_create("deconv/W", shape_w, w_init);

  if (with_bias) {
    if (b_init == nullptr) {
      shared_b_init = make_shared<ConstantInitializer>();
      b_init = shared_b_init.get();
    }

    CgVariablePtr deconv_b =
        parameters.get_parameter_or_create("deconv/b", {n_map_out}, b_init);

    bool execute_forward = true;
    return connect(make_shared<CgFunction>(create_Deconvolution(
                       ctx, base_axis, pad, stride, dilation, group)),
                   {x, deconv_w, deconv_b}, execute_forward);

  } else {

    bool execute_forward = true;
    return connect(make_shared<CgFunction>(create_Deconvolution(
                       ctx, base_axis, pad, stride, dilation, group)),
                   {x, deconv_w}, execute_forward);
  }
}

CgVariablePtr deconvolution(CgVariablePtr x, int base_axis, int n_map_out,
                            const vector<int> &kernel,
                            ParameterDirectory parameters,
                            ConvolutionOpts conv_opts) {
  auto global_ctx =
      SingletonManager::get<GlobalContext>()->get_current_context();
  return deconvolution(global_ctx, x, base_axis, n_map_out, kernel,
                       conv_opts.pad(), conv_opts.stride(),
                       conv_opts.dilation(), conv_opts.group(), parameters,
                       conv_opts.with_bias(), conv_opts.fix_parameters(),
                       conv_opts.w_init(), conv_opts.b_init())[0];
}

vector<CgVariablePtr>
batch_normalization(Context &ctx, CgVariablePtr x, const vector<int> &axes,
                    float decay_rate, float eps, bool batch_stat,
                    ParameterDirectory parameters, bool fix_parameters) {

  NBLA_CHECK(axes.size() == 1, error_code::value, "Size of axes should be 1");
  Shape_t shape_stat = x->variable()->shape();
  for (int i = 0; i < shape_stat.size(); i++)
    if (i != axes[0])
      shape_stat[i] = 1;

  ConstantInitializer beta_init(0.0);
  ConstantInitializer gamma_init(1.0);
  ConstantInitializer mean_init(0.0);
  ConstantInitializer variance_init(1.0);

  CgVariablePtr beta =
      parameters.get_parameter_or_create("bn/beta", shape_stat, &beta_init);
  CgVariablePtr gamma =
      parameters.get_parameter_or_create("bn/gamma", shape_stat, &gamma_init);
  CgVariablePtr mean = parameters.get_parameter_or_create("bn/mean", shape_stat,
                                                          &mean_init, false);
  CgVariablePtr variance = parameters.get_parameter_or_create(
      "bn/variance", shape_stat, &variance_init, false);

  bool execute_forward = true;
  return connect(make_shared<CgFunction>(create_BatchNormalization(
                     ctx, axes, decay_rate, eps, batch_stat)),
                 {x, beta, gamma, mean, variance}, execute_forward);
}

CgVariablePtr batch_normalization(CgVariablePtr x, bool batch_stat,
                                  ParameterDirectory parameters,
                                  BatchNormalizationOpts batch_opts) {
  auto global_ctx =
      SingletonManager::get<GlobalContext>()->get_current_context();
  return batch_normalization(
      global_ctx, x, batch_opts.axes(), batch_opts.decay_rate(),
      batch_opts.eps(), batch_stat, parameters, batch_opts.fix_parameters())[0];
}
}
}
