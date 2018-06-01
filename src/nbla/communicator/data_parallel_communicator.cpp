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

#include <nbla/communicator/data_parallel_communicator.hpp>

#include <algorithm>
#include <memory>

namespace nbla {

// TODO: should separate interface for each communicator, perhaps
// like solver does.
NBLA_REGISTER_COMMUNICATOR_SOURCE(DataParallelCommunicator);

using std::make_shared;

template <typename T>
DataParallelCommunicator<T>::DataParallelCommunicator(const Context &ctx)
    : Communicator(ctx) {}

template <typename T>
DataParallelCommunicator<T>::~DataParallelCommunicator() {}

template <typename T> void DataParallelCommunicator<T>::init() {
  NBLA_ERROR(error_code::not_implemented, "CPU init is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::reduce(const vector<NdArrayPtr> &ndarray_list,
                                         int dst, bool division, bool inplace,
                                         const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU reduce is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::reduce(NdArrayPtr ndarray, int dst,
                                         bool division, bool inplace,
                                         const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU reduce is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::allreduce(bool division, bool inplace) {
  NBLA_ERROR(error_code::not_implemented, "CPU allreduce is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::all_reduce(
    const vector<NdArrayPtr> &ndarray_list, bool division, bool inplace,
    const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU all_reduce is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::all_reduce(NdArrayPtr ndarray, bool division,
                                             bool inplace,
                                             const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU all_reduce is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::reduce_scatter(
    const vector<NdArrayPtr> &ndarray_list, NdArrayPtr ndarray, bool division,
    const string &group) {
  NBLA_ERROR(error_code::not_implemented,
             "CPU reduce_scatter is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::bcast(const vector<NdArrayPtr> &ndarray_list,
                                        int src, bool inplace,
                                        const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU bcast is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::bcast(NdArrayPtr ndarray, int src,
                                        bool inplace, const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU bcast is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::all_gather(
    NdArrayPtr ndarray, const vector<NdArrayPtr> &ndarray_list,
    const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CPU all_gather is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::reduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
             "CPU reduce_async is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::allreduce_async(bool division, bool inplace) {
  NBLA_ERROR(error_code::not_implemented,
             "CPU allreduce_async is not implemented.")
}

template <typename T>
void DataParallelCommunicator<T>::reducescatter_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
             "CPU reducescatter_async is not implemented.")
}

template <typename T> void DataParallelCommunicator<T>::bcast_async() {
  NBLA_ERROR(error_code::not_implemented, "CPU bcast_async is not implemented.")
}

template <typename T> void DataParallelCommunicator<T>::allgather_async() {
  NBLA_ERROR(error_code::not_implemented,
             "CPU allgather_async is not implemented.")
}

template <typename T>
vector<string> DataParallelCommunicator<T>::allowed_array_classes() {
  NBLA_ERROR(error_code::not_implemented, "Derived class of "
                                          "DataParallelCommunicator must "
                                          "implement allowed_array_classes().")
}

template class DataParallelCommunicator<float>;

template class DataParallelCommunicator<Half>;
}
