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

#include <nbla/communicator.hpp>
#include <nbla/logger.hpp>

#include <algorithm>
#include <memory>

namespace nbla {

using std::make_shared;

Communicator::Communicator(const Context &ctx) : ctx_(ctx) {}

Communicator::~Communicator() {}

void Communicator::add_context_and_parameters(
    const pair<Context, vector<pair<string, VariablePtr> > > &ctx_params) {
  auto ctx0 = ctx_params.first;
  auto identifier0 = get_array_key_from_context(ctx0);
  for (auto ctx : contexts_) {
    auto identifier = get_array_key_from_context(ctx);
    if (identifier0 == identifier) {
      NBLA_ERROR(error_code::value,
          "The same context was added.")
    }
  }
  contexts_.push_back(ctx_params.first);
  device_func_named_param_.push_back(ctx_params.second);
}

void Communicator::remove_context_parameters(
    const pair<Context, vector<string>> &keys) {
  NBLA_ERROR(error_code::not_implemented,
      "clear_context_parameters not implemented")
}

void Communicator::clear_context_parameters() {
  NBLA_ERROR(error_code::not_implemented,
      "clear_context_parameters not implemented")
}

void Communicator::init() {
  if (initialized_) {
    NBLA_ERROR(error_code::value,
      "Communicator was already initialized.")
  }
}

void Communicator::check_array_class(Context ctx, VariablePtr vp) {
  auto ctx_array_class = ctx.array_class;
  auto array_class = vp->grad()->array()->head_array_class();


  if (ctx_array_class != array_class) {
    NBLA_LOG_WARN(
        "\n"
        "###################################################################\n"
        "Data are on different devices. Collective operations in　\n"
        "`communicator` can be called for data which are on the same device.\n"
        "Please CHECK the function implementation of the device \n"
        "now you are calling. For example, Affine CUDA might not be \n"
        "implemented.\n"
        "###################################################################\n"
        )
  }
}

void Communicator::reduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU reduce is not implemented.")
}

void Communicator::allreduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU allreduce is not implemented.")
}

void Communicator::reducescatter(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU reducescatter is not implemented.")
}

void Communicator::bcast() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU bcast is not implemented.")
}

void Communicator::allgather() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU allgather is not implemented.")
}

void Communicator::ireduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ireduce is not implemented.")
}

void Communicator::iallreduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU iallreduce is not implemented.")
}

void Communicator::ireducescatter(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ireducescatter is not implemented.")
}

void Communicator::ibcast() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ibcast is not implemented.")
}

void Communicator::iallgather() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU iallgather is not implemented.")
}

void Communicator::reduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU reduce_async is not implemented.")
}

void Communicator::allreduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU allreduce_async is not implemented.")
}

void Communicator::reducescatter_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU reducescatter_async is not implemented.")
}

void Communicator::bcast_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU bcast_async is not implemented.")
}

void Communicator::allgather_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU allgather_async is not implemented.")
}

void Communicator::ireduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ireduce_async is not implemented.")
}

void Communicator::iallreduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU iallreduce_async is not implemented.")
}

void Communicator::ireducescatter_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ireducescatter_async is not implemented.")
}

void Communicator::ibcast_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ibcast_async is not implemented.")
}

void Communicator::iallgather_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU iallgather_async is not implemented.")
}

vector<string> Communicator::allowed_array_classes() {
  NBLA_ERROR(error_code::not_implemented,
      "Derived class of Communicator must implement allowed_array_classes().")
}
}
