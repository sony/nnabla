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
    const pair<Context, vector<pair<string, VariablePtr>>> &ctx_params) {
  auto ctx0 = ctx_params.first;
  auto identifier0 = get_array_key_from_context(ctx0);
  for (auto ctx : contexts_) {
    auto identifier = get_array_key_from_context(ctx);
    if (identifier0 == identifier) {
      NBLA_ERROR(error_code::value, "The same context was added.")
    }
  }
  contexts_.push_back(ctx_params.first);
  device_func_named_param_.push_back(ctx_params.second);

  // Count number of parameters in total.
  Size_t sum_ = 0;
  for (auto e : ctx_params.second) {
    sum_ += e.second->size();
  }
  total_params_ = sum_;
}

void Communicator::barrier() {
  NBLA_ERROR(error_code::not_implemented, "Barrier is not implemented in CPU.")
}

void Communicator::abort() {
  NBLA_ERROR(error_code::not_implemented, "Abort is not implemented in CPU.")
}

int Communicator::rank() { return rank_; }

int Communicator::local_rank() { return local_rank_; }

int Communicator::size() { return size_; }

void Communicator::remove_context_parameters(
    const pair<Context, vector<string>> &keys) {
  NBLA_ERROR(error_code::not_implemented,
             "clear_context_parameters not implemented")
}

void Communicator::clear_context_parameters() {
  total_params_ = 0;
  vector<Context>().swap(contexts_);
  decltype(device_func_named_param_)().swap(device_func_named_param_);
  decltype(func_device_named_param_)().swap(func_device_named_param_);
}

void Communicator::init() {
  if (initialized_) {
    NBLA_ERROR(error_code::value, "Communicator was already initialized.")
  }
}

void Communicator::check_array_class(Context ctx, VariablePtr vp) {
  auto ctx_array_class = ctx.array_class;
  auto array_class = vp->grad()->array()->head_array_class();

  if (ctx_array_class != array_class) {
    NBLA_LOG_WARN(
        "\n"
        "###################################################################\n"
        "Data are on different devices. Collective operations in \n"
        "`communicator` can be called for data which are on the same device.\n"
        "Please CHECK the function implementation of the device \n"
        "now you are calling. For example, Affine CUDA might not be \n"
        "implemented.\n"
        "###################################################################\n")
  }
}

string Communicator::new_group(pair<string, vector<int>> name_ranks_pair) {
  NBLA_ERROR(error_code::not_implemented, "CPU new_group is not implemented.")
}

unordered_map<string, vector<int>> Communicator::list_groups() {
  NBLA_ERROR(error_code::not_implemented, "CPU list_group is not implemented.")
}

vector<int> Communicator::find_group(const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU list_group is not implemented.")
}

void Communicator::reduce(const vector<NdArrayPtr> &ndarray_list, int dst,
                          bool division, bool inplace, const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU reduce is not implemented.")
}

void Communicator::reduce(NdArrayPtr ndarray, int dst, bool division,
                          bool inplace, const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU reduce is not implemented.")
}

void Communicator::allreduce(bool division, bool inplace) {
  NBLA_ERROR(error_code::not_implemented, "CPU allreduce is not implemented.")
}

void Communicator::all_reduce(const vector<NdArrayPtr> &ndarray_list,
                              bool division, bool inplace,
                              const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU all_reduce is not implemented.")
}

void Communicator::all_reduce(NdArrayPtr ndarray, bool division, bool inplace,
                              const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU all_reduce is not implemented.")
}

CommunicatorBackwardCallbackPtr
Communicator::all_reduce_callback(const vector<NdArrayPtr> &ndarray_list,
                                  size_t pack_size, bool division,
                                  const string &group) {
  NBLA_ERROR(error_code::not_implemented,
             "CPU all_reduce_callback is not implemented")
}
CommunicatorBackwardCallbackPtr
Communicator::all_reduce_callback(NdArrayPtr ndarray, size_t pack_size,
                                  bool division, const string &group) {
  return this->all_reduce_callback(vector<NdArrayPtr>{ndarray}, pack_size,
                                   division, group);
}

void Communicator::reduce_scatter(const vector<NdArrayPtr> &ndarray_list,
                                  NdArrayPtr ndarray, bool division,
                                  const string &group) {
  NBLA_ERROR(error_code::not_implemented,
             "CPU reduce_scatter is not implemented.")
}

void Communicator::bcast(const vector<NdArrayPtr> &ndarray_list, int src,
                         bool inplace, const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU bcast is not implemented.")
}

void Communicator::bcast(NdArrayPtr ndarray, int src, bool inplace,
                         const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU bcast is not implemented.")
}

void Communicator::all_gather(NdArrayPtr ndarray,
                              const vector<NdArrayPtr> &ndarray_list,
                              const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU all_gather is not implemented.")
}

void Communicator::reduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
             "CPU reduce_async is not implemented.")
}

void Communicator::allreduce_async(bool division, bool inplace) {
  NBLA_ERROR(error_code::not_implemented,
             "CPU allreduce_async is not implemented.")
}

void Communicator::reducescatter_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
             "CPU reducescatter_async is not implemented.")
}

void Communicator::bcast_async() {
  NBLA_ERROR(error_code::not_implemented, "CPU bcast_async is not implemented.")
}

void Communicator::allgather_async() {
  NBLA_ERROR(error_code::not_implemented,
             "CPU allgather_async is not implemented.")
}

vector<string> Communicator::allowed_array_classes() {
  NBLA_ERROR(
      error_code::not_implemented,
      "Derived class of Communicator must implement allowed_array_classes().")
}
}
