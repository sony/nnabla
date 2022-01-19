// Copyright 2021 Sony Group Corporation.
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

#include <nbla/function/utils/dropout_workaround.hpp>

namespace nbla {

VariablePtr get_dropout_mask(VariablePtr dropout_input) {
  NBLA_CHECK(dropout_input->dropout_mask_, error_code::unclassified,
             "Set a mask by Dropout::setup before.");
  return dropout_input->dropout_mask_;
}

void set_dropout_mask(Variable *dropout_input, VariablePtr dropout_mask) {
  dropout_input->dropout_mask_ = dropout_mask;
}
}
