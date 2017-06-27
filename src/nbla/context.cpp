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

#include <nbla/context.hpp>

namespace nbla {
Context::Context(const string &backend, const string &array_class,
                 const string &device_id, const string &compute_backend)
    : backend(backend), array_class(array_class), device_id(device_id),
      compute_backend(compute_backend) {}
Context &Context::set_backend(const string &backend) {
  this->backend = backend;
  return *this;
}
Context &Context::set_array_class(const string &array_class) {
  this->array_class = array_class;
  return *this;
}
Context &Context::set_device_id(const string &device_id) {
  this->device_id = device_id;
  return *this;
}
Context &Context::set_compute_backend(const string &compute_backend) {
  this->compute_backend = compute_backend;
  return *this;
}
}
