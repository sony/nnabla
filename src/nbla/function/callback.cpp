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
}
