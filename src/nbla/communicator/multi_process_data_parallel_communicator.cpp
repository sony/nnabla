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

#include <nbla/communicator/multi_process_data_parallel_communicator.hpp>

#include <algorithm>
#include <memory>

namespace nbla {

//TODO: should separate interface for each communicator, perhaps
// like solver does.
NBLA_REGISTER_COMMUNICATOR_SOURCE(MultiProcessDataParallelCommunicator);

using std::make_shared;

template<typename T>
MultiProcessDataParallelCommunicator<T>::MultiProcessDataParallelCommunicator(const Context &ctx) : Communicator(ctx) {}

template<typename T>
MultiProcessDataParallelCommunicator<T>::~MultiProcessDataParallelCommunicator() {}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::init() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU init is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::reduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU reduce is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::allreduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU allreduce is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::reducescatter(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU reducescatter is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::bcast() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU bcast is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::allgather() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU allgather is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::ireduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ireduce is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::iallreduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU iallreduce is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::ireducescatter(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ireducescatter is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::ibcast() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ibcast is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::iallgather() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU iallgather is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::reduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU reduce_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::allreduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU allreduce_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::reducescatter_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU reducescatter_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::bcast_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU bcast_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::allgather_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU allgather_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::ireduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ireduce_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::iallreduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU iallreduce_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::ireducescatter_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ireducescatter_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::ibcast_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ibcast_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicator<T>::iallgather_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU iallgather_async is not implemented.")
}

template<typename T>
vector<string> MultiProcessDataParallelCommunicator<T>::allowed_array_classes() {
  NBLA_ERROR(error_code::not_implemented,
      "Derived class of MultiProcessDataParallelCommunicator must implement allowed_array_classes().")
}

template class MultiProcessDataParallelCommunicator<float>;
}
