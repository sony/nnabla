// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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

#include "nnp_impl.hpp"
#include <iostream>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/function/mean.hpp>
#include <nbla/function/sink.hpp>
#include <random>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define open _open
#define O_RDONLY _O_RDONLY
#endif

namespace nbla {
namespace utils {
namespace nnp {

// ----------------------------------------------------------------------
// OptimizerImpl
// ----------------------------------------------------------------------
const nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

OptimizerImpl::OptimizerImpl(const nbla::Context &ctx,
                             const ::Optimizer &optimizer,
                             shared_ptr<Network> network,
                             shared_ptr<DatasetImpl> dataset)
    : ctx_(ctx), optimizer_proto_(optimizer), network_(network),
      dataset_(dataset) {

  solver_ = create_solver(optimizer.solver());
  vector<pair<string, VariablePtr>> params;
  for (auto it = optimizer_proto_.parameter_variable().begin();
       it != optimizer_proto_.parameter_variable().end(); it++) {
    const string name = it->variable_name();
    const float lr_value = it->learning_rate_multiplier();
    const CgVariablePtr variable = network_->get_variable(it->variable_name());
    if (lr_value != 0.0) {
      variable->set_need_grad(true);
      params.push_back({name, variable->variable()});
    } else {
      variable->set_need_grad(false);
    }
  }
  solver_->set_parameters(params, /*reset*/ true, /*retain_state*/ false);
  if (update_interval() != 1) {
    std::cout << "Update interval is forced to 1." << std::endl;
  }

  data_iterator_ = shared_ptr<DataIteratorFromCacheFiles>(
      new DataIteratorFromCacheFiles(dataset));
  network_->set_batch_size(data_iterator_->get_batch_size());
}

string OptimizerImpl::name() const { return optimizer_proto_.name(); }
string OptimizerImpl::network_name() const {
  return optimizer_proto_.network_name();
}
string OptimizerImpl::dataset_name() const {
  if (optimizer_proto_.dataset_name_size() != 1) {
    NBLA_ERROR(error_code::value, "Currently only one dataset supported.");
  }
  return optimizer_proto_.dataset_name()[0];
}
const int OptimizerImpl::update_interval() const {
  return optimizer_proto_.update_interval();
}

vector<Optimizer::DataVariable> OptimizerImpl::get_data_variables() {
  vector<Optimizer::DataVariable> ret;
  for (auto it = optimizer_proto_.data_variable().begin();
       it != optimizer_proto_.data_variable().end(); it++) {
    Optimizer::DataVariable v{it->variable_name(), it->data_name(),
                              network_->get_variable(it->variable_name())};
    ret.push_back(v);
  }
  return ret;
}

vector<Optimizer::GeneratorVariable> OptimizerImpl::get_generator_variables() {
  vector<Optimizer::GeneratorVariable> ret;
  for (auto it = optimizer_proto_.generator_variable().begin();
       it != optimizer_proto_.generator_variable().end(); it++) {
    Optimizer::GeneratorVariable v{it->variable_name(), it->type(),
                                   it->multiplier(),
                                   network_->get_variable(it->variable_name())};
    ret.push_back(v);
  }
  return ret;
}

vector<Optimizer::LossVariable> OptimizerImpl::get_loss_variables() {
  vector<Optimizer::LossVariable> ret;
  for (auto it = optimizer_proto_.loss_variable().begin();
       it != optimizer_proto_.loss_variable().end(); it++) {
    Optimizer::LossVariable v{it->variable_name(),
                              network_->get_variable(it->variable_name())};
    ret.push_back(v);
  }
  NBLA_CHECK(ret.size() > 0, error_code::value,
             "Optimizer `%s`'s loss is empty.", name().c_str());
  return ret;
}

vector<Optimizer::ParameterVariable> OptimizerImpl::get_parameter_variables() {
  vector<Optimizer::ParameterVariable> ret;
  for (auto it = optimizer_proto_.parameter_variable().begin();
       it != optimizer_proto_.parameter_variable().end(); it++) {
    Optimizer::ParameterVariable v{it->variable_name(),
                                   it->learning_rate_multiplier(),
                                   network_->get_variable(it->variable_name())};
    ret.push_back(v);
  }
  return ret;
}

shared_ptr<Network> OptimizerImpl::get_network() { return network_; }

void OptimizerImpl::set_parameters(
    const vector<Optimizer::ParameterVariable> &params, bool reset,
    bool retain_state) {
  vector<pair<string, VariablePtr>> parameters;
  for (auto param : params) {
    const string name = param.variable_name;
    const float lr_value = param.learning_rate_multiplier;
    const CgVariablePtr variable = param.variable;
    if (lr_value != 0.0) {
      variable->set_need_grad(true);
      parameters.push_back({name, variable->variable()});
    } else {
      variable->set_need_grad(false);
    }
  }
  solver_->set_parameters(parameters, reset, retain_state);
}

string OptimizerImpl::type() const { return optimizer_proto_.solver().type(); }

float OptimizerImpl::weight_decay_rate() const {
  return optimizer_proto_.solver().weight_decay();
}

float OptimizerImpl::lr_decay() const {
  return optimizer_proto_.solver().lr_decay();
}

long long int OptimizerImpl::lr_decay_interval() const {
  return optimizer_proto_.solver().lr_decay_interval();
}

void OptimizerImpl::zero_grad() { solver_->zero_grad(); }

void OptimizerImpl::update_parameters() { solver_->update(); }

float OptimizerImpl::learning_rate() const { return solver_->learning_rate(); }

void OptimizerImpl::set_learning_rate(float learning_rate) {
  solver_->set_learning_rate(learning_rate);
}

void OptimizerImpl::weight_decay(float decay_rate) {
  solver_->weight_decay(decay_rate);
}

void OptimizerImpl::remove_parameters(const vector<string> &keys) {
  solver_->remove_parameters(keys);
}

void OptimizerImpl::clear_parameters() { solver_->clear_parameters(); }

std::random_device seed_gen;
std::default_random_engine engine(seed_gen());
std::uniform_real_distribution<> uniform(0.0, 1.0);
std::normal_distribution<> normal(0.0, 1.0);

const float OptimizerImpl::update(const int iter) {

  auto minibatch = data_iterator_->next();
  for (auto x : this->get_data_variables()) {
    VariablePtr v = x.variable->variable();
    v->set_data(minibatch.at(x.data_name));
  }

  for (auto x : this->get_generator_variables()) {
    VariablePtr v = x.variable->variable();
    float_t *generator = v->cast_data_and_get_pointer<float_t>(cpu_ctx);
    if (x.type == "Normal") {
      for (int i = 0; i < v->size(); i++)
        generator[i] = x.multiplier * normal(engine);
    } else if (x.type == "Uniform") {
      for (int i = 0; i < v->size(); i++)
        generator[i] = x.multiplier * uniform(engine);
    } else if (x.type == "Constant") {
      for (int i = 0; i < v->size(); i++)
        generator[i] = x.multiplier;
    }
  }

  if (this->get_loss_variables().size() <= 0) {
    return 0.0;
  }

  vector<CgVariablePtr> l_means;
  for (auto loss : this->get_loss_variables()) {
    const int ndim = loss.variable->variable()->ndim();
    vector<int> axes;
    for (int i = 0; i < ndim; i++)
      axes.push_back(i);
    auto mean = make_shared<CgFunction>(create_Mean(ctx_, axes, false));
    l_means.push_back(connect(mean, {loss.variable}, 1)[0]);
  }
  auto sink = make_shared<CgFunction>(create_Sink(ctx_, true));
  CgVariablePtr l_sink = connect(sink, l_means, 1)[0];

  this->zero_grad();
  l_sink->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
  l_sink->backward();
  ///*NdArrayPtr grad=*/nullptr, /*clear_buffer=*/true);
  this->weight_decay(this->weight_decay_rate());
  this->update_parameters();

  if ((iter + 1) % this->lr_decay_interval()) {
    this->set_learning_rate(this->learning_rate() * this->lr_decay());
  }

  float cost = 0.0;
  for (auto l_mean : l_means) {
    float_t *values =
        l_mean->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx);
    cost += values[0];
  }
  return cost;
}
}
}
}
