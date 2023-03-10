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

#include <nbla/function/callback.hpp>

namespace nbla {

void Callback::setup_impl(const Variables &inputs, const Variables &outputs) {
  setup_callback_(obj_, inputs, outputs);
}

void Callback::forward_impl(const Variables &inputs, const Variables &outputs) {
  forward_callback_(obj_, inputs, outputs);
}

void Callback::backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) {
  backward_callback_(obj_, inputs, outputs, propagate_down, accum);
}

vector<string> Callback::allowed_array_classes() {
  return SingletonManager::get<Cpu>()->array_classes();
}

NBLA_API shared_ptr<Function> create_Callback(
    const Context &ctx, void *obj, int min_outputs,
    Callback::setup_callback_type s, Callback::forward_callback_type f,
    Callback::backward_callback_type b, Callback::cleanup_callback_type c,
    Callback::grad_depends_output_data_callback_type go,
    Callback::grad_depends_input_data_callback_type gi) {
  return make_shared<Callback>(ctx, obj, min_outputs, s, f, b, c, go, gi);
}
}
