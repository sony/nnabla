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

#include <nbla/cpu.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/solver.hpp>

#include <algorithm>
#include <memory>

namespace nbla {

using std::make_shared;

Solver::Solver(const Context &ctx) : ctx_(ctx), setup_called_(false) {}

Solver::~Solver() {}

void Solver::setup() {
  if (setup_called_)
    return;
  // Check if specifiedd array_class by context matches to allowed array
  // classes.
  int array_class_index =
      0; // Default array is 0-th array_class in allowed_array_classes().
  for (int i = 0; i < this->allowed_array_classes().size(); ++i) {
    if (ctx_.array_class == this->allowed_array_classes()[i]) {
      array_class_index = i;
    }
  }
  ctx_.set_array_class(this->allowed_array_classes()[array_class_index]);
  setup_called_ = true;
}

void Solver::set_parameters(const vector<pair<string, VariablePtr>> &params,
                            bool reset, bool retain_state) {
  setup();
  if (reset) {
    clear_parameters();
  }
  for (auto &kv : params) {
    if (!reset) {
      auto it = params_.find(kv.first);
      if (it != params_.end()) {
        // Parameter found.
        if (retain_state) {
          NBLA_CHECK(kv.second->shape() == it->second.p->shape(),
                     error_code::value,
                     "Could ont retain state. The shapes of %s didn't match. "
                     "Given: (%s) != previously: (%s)",
                     kv.first.c_str(),
                     string_join(kv.second->shape(), string(", ")).c_str(),
                     string_join(it->second.p->shape(), string(", ")).c_str());
          continue;
        }
        remove_state_impl(kv.first);
      }
    }
    params_.insert({kv.first, {kv.second, 0}});
    set_state_impl(kv.first, kv.second);
  }
}

void Solver::remove_parameters(const vector<string> &keys) {
  for (auto &key : keys) {
    params_.erase(key);
    remove_state_impl(key);
  }
}

void Solver::clear_parameters() {
  for (auto &kv : params_) {
    auto &key = kv.first;
    remove_state_impl(key);
  }
  params_.clear();
}

void Solver::zero_grad() {
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    g->zero();
    kv.second.at = g->modification_count();
  }
}

void Solver::update() {
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (kv.second.at == g->modification_count()) {
      // The gradient is not computed. Skip.
      continue;
    }
    update_impl(kv.first, kv.second.p);
  }
}

void Solver::weight_decay(float decay_rate) {
  if (decay_rate == 0)
    return;
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (kv.second.at == g->modification_count()) {
      // The gradient is not computed. Skip.
      continue;
    }
    weight_decay_impl(kv.first, kv.second.p, decay_rate);
  }
}

bool Solver::check_inf_grad() {
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (kv.second.at == g->modification_count()) {
      // The gradient is not computed. Skip.
      continue;
    }
    if (check_inf_grad_impl(kv.first, kv.second.p)) {
      return true;
    }
  }
  return false;
}

// TODO: potential to speed-up
bool Solver::check_nan_grad() {
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (kv.second.at == g->modification_count()) {
      // The gradient is not computed. Skip.
      continue;
    }
    if (check_nan_grad_impl(kv.first, kv.second.p)) {
      return true;
    }
  }
  return false;
}

// TODO: potential to speed-up
bool Solver::check_inf_or_nan_grad() {
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (kv.second.at == g->modification_count()) {
      // The gradient is not computed. Skip.
      continue;
    }
    if (check_inf_or_nan_grad_impl(kv.first, kv.second.p)) {
      return true;
    }
  }
  return false;
}

// Methods for the mixed-precision training
void Solver::scale_grad(float scale) {
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (kv.second.at == g->modification_count()) {
      // The gradient is not computed. Skip.
      continue;
    }
    scale_grad_impl(kv.first, kv.second.p, scale);
  }
}

vector<string> Solver::allowed_array_classes() {
  return SingletonManager::get<Cpu>()->array_classes();
}
}
