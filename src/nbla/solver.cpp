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
#include <nbla/global_solver_callback.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/solver.hpp>

#include <algorithm>
#include <memory>

// Should be false, unless you want to executing larger model than allotted
// memory with compromising computational accuracy.
#define CLEAR_MEMORY false

namespace nbla {

#if CLEAR_MEMORY
void clear_param(pair<string, Params> &kv) {
  // For reusing memory with comprimising computational accuracy (mainly for
  // executing larger model than alloted memory).
  kv.second.p->data()->array()->clear();
  kv.second.p->grad()->array()->clear();
  clear_state(kv.first);
}
#endif

using std::make_shared;
using std::make_pair;

/** UpdateHookWithObject Implementation **/
UpdateHookWithObject::UpdateHookWithObject() {}

UpdateHookWithObject::UpdateHookWithObject(const UpdateHookWithObject &from)
    : obj_(from.obj_), callback_(from.callback_),
      setup_callback_(from.setup_callback_),
      cleanup_callback_(from.cleanup_callback_) {
  setup_callback_(obj_);
}

UpdateHookWithObject::UpdateHookWithObject(void *obj, callback_type cb,
                                           setup_callback_type setup_cb,
                                           cleanup_callback_type cleanup_cb)
    : obj_(obj), callback_(cb), setup_callback_(setup_cb),
      cleanup_callback_(cleanup_cb) {
  setup_callback_(obj_);
}

UpdateHookWithObject::~UpdateHookWithObject() { cleanup_callback_(obj_); }

UpdateHookWithObject &UpdateHookWithObject::
operator=(const UpdateHookWithObject &rhs) {
  // check self-assignment
  if (&rhs == this)
    return *this;

  obj_ = rhs.obj_;
  callback_ = rhs.callback_;
  setup_callback_ = rhs.setup_callback_;
  cleanup_callback_ = rhs.cleanup_callback_;

  setup_callback_(obj_);

  return *this;
}

void UpdateHookWithObject::operator()() { callback_(obj_); }

/** Solver Implementation**/
Solver::Solver(const Context &ctx) : ctx_(ctx), setup_called_(false) {}

Solver::~Solver() {}

void Solver::setup() {
  if (setup_called_)
    return;
  // Check if specified array_class by context matches to allowed array
  // classes.
  int array_class_index =
      0; // Default array is 0-th array_class in allowed_array_classes().
  for (unsigned int i = 0; i < this->allowed_array_classes().size(); ++i) {
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
                     "Could not retain state. The shapes of %s didn't match. "
                     "Given: (%s) != previously: (%s)",
                     kv.first.c_str(),
                     string_join(kv.second->shape(), string(", ")).c_str(),
                     string_join(it->second.p->shape(), string(", ")).c_str());
          continue;
        }
        remove_state_impl(kv.first);
      }
    }
    params_.insert({kv.first, {kv.second}});
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

vector<pair<string, VariablePtr>> Solver::get_parameters() {
  vector<pair<string, VariablePtr>> params;
  for (auto &kv : params_) {
    auto elm = make_pair(kv.first, kv.second.p);
    params.push_back(elm);
  }
  return params;
}

vector<pair<string, Solver::SolverState>> Solver::get_states() {
  vector<pair<string, Solver::SolverState>> states;
  for (auto &kv0 : states_) {
    states.push_back({kv0.first, kv0.second});
  }
  return states;
}

void Solver::set_states(const vector<pair<string, SolverState>> &states) {
  for (auto &kv0 : states) {
    auto it = states_.find(kv0.first);
    NBLA_CHECK(it != states_.end(), error_code::value,
               "Set weight parameter for %s first.", kv0.first.c_str());
    it->second = kv0.second;
  }
}

void Solver::zero_grad() {
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    g->zero();
  }
}

namespace {

struct ScopedCallback {
  update_hook_type post_;
  ScopedCallback(update_hook_type &pre, update_hook_type &post) : post_(post) {
    // Call global pre_hooks first
    SingletonManager::get<GlobalSolverCallback>()->call_pre_hooks();

    if (pre) {
      pre();
    }
  }
  ~ScopedCallback() {
    if (post_) {
      post_();
    }

    // Call global post_hooks last
    SingletonManager::get<GlobalSolverCallback>()->call_post_hooks();
  }
};
}

void Solver::update(update_hook_type pre_callback,
                    update_hook_type post_callback) {

  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (g->zeroing()) {
      // The gradient is not computed. Skip.
      continue;
    }
    ScopedCallback(pre_callback, post_callback);
    update_impl(kv.first, kv.second.p);

#if CLEAR_MEMORY
    clear_param(kv);
#endif
  }
}

void Solver::weight_decay(float decay_rate, update_hook_type pre_callback,
                          update_hook_type post_callback) {
  if (decay_rate == 0)
    return;
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (g->zeroing()) {
      // The gradient is not computed. Skip.
      continue;
    }
    ScopedCallback(pre_callback, post_callback);
    weight_decay_impl(kv.first, kv.second.p, decay_rate);

#if CLEAR_MEMORY
    clear_param(kv);
#endif
  }
}

void Solver::clip_grad_by_norm(float norm, update_hook_type pre_callback,
                               update_hook_type post_callback) {
  if (norm == 0)
    return;
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (g->zeroing()) {
      // The gradient is not computed. Skip.
      continue;
    }
    ScopedCallback(pre_callback, post_callback);
    clip_grad_by_norm_impl(kv.first, kv.second.p, norm);

#if CLEAR_MEMORY
    clear_param(kv);
#endif
  }
}

bool Solver::check_inf_grad(update_hook_type pre_callback,
                            update_hook_type post_callback) {
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (g->zeroing()) {
      // The gradient is not computed. Skip.
      continue;
    }
    ScopedCallback(pre_callback, post_callback);
    if (check_inf_grad_impl(kv.first, kv.second.p)) {
      return true;
    }
  }
  return false;
}

// TODO: potential to speed-up
bool Solver::check_nan_grad(update_hook_type pre_callback,
                            update_hook_type post_callback) {
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (g->zeroing()) {
      // The gradient is not computed. Skip.
      continue;
    }
    ScopedCallback(pre_callback, post_callback);
    if (check_nan_grad_impl(kv.first, kv.second.p)) {
      return true;
    }
  }
  return false;
}

// TODO: potential to speed-up
bool Solver::check_inf_or_nan_grad(update_hook_type pre_callback,
                                   update_hook_type post_callback) {
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (g->zeroing()) {
      // The gradient is not computed. Skip.
      continue;
    }
    ScopedCallback(pre_callback, post_callback);
    if (check_inf_or_nan_grad_impl(kv.first, kv.second.p)) {
      return true;
    }
  }
  return false;
}

// Methods for the mixed-precision training
void Solver::scale_grad(float scale, update_hook_type pre_callback,
                        update_hook_type post_callback) {
  for (auto &kv : params_) {
    SyncedArrayPtr g = kv.second.p->grad()->array();
    if (g->zeroing()) {
      // The gradient is not computed. Skip.
      continue;
    }
    ScopedCallback(pre_callback, post_callback);
    scale_grad_impl(kv.first, kv.second.p, scale);

#if CLEAR_MEMORY
    clear_param(kv);
#endif
  }
}

vector<string> Solver::allowed_array_classes() {
  return SingletonManager::get<Cpu>()->array_classes();
}
}
