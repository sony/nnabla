// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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

#ifndef __NBLA_UTILS_ND_INDEX_HPP__
#define __NBLA_UTILS_ND_INDEX_HPP__

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <numeric>
#include <sstream>
#include <vector>

namespace nbla {
namespace ndi {

template <typename T> std::string str(const std::vector<T> &index) {
  std::ostringstream os;
  os << "[";
  auto print = [&os](size_t i) { os << i << ", "; };
  std::for_each(index.begin(), index.end() - 1, print);
  os << index.back() << ']';
  return os.str();
}

template <typename T>
inline std::vector<T> strides(const std::vector<T> &shape) {
  std::vector<T> v(shape.size(), 1);
  std::copy(shape.begin() + 1, shape.end(), v.begin());
  std::partial_sum(v.rbegin(), v.rend(), v.rbegin(), std::multiplies<T>());
  return v;
}

template <typename T>
inline T nd2flat(const std::vector<T> &index, const std::vector<T> &stride) {
  assert(index.size() <= stride.size());
  return std::inner_product(index.begin(), index.end(), stride.begin(), 0);
}

template <typename T>
inline T nd2flat(const std::vector<T> &index, const std::vector<T> &stride,
                 int axis) {
  using std::inner_product;
  axis = axis < 0 ? stride.size() + axis : axis;
  assert(0 <= axis && axis < index.size());
  assert(index.size() <= stride.size());
  return inner_product(index.begin(), index.begin() + axis, stride.begin(), 0);
}

template <typename T>
inline T nd2flat(const std::vector<T> &index, const std::vector<T> &stride,
                 const std::pair<int, int> &axis) {
  int axis_from = axis.first < 0 ? stride.size() + axis.first : axis.first;
  int axis_last = axis.second < 0 ? stride.size() + axis.second : axis.second;
  assert(0 <= axis_from && axis_from <= axis_last && axis_last < index.size());
  assert(index.size() <= stride.size());
  assert(index.begin() + axis_last + 1 == index.end());
  T result = 0;
  for (; axis_from < axis_last; ++axis_from) {
    result += index[axis_from] * stride[axis_from];
  }
  return result;
}

template <typename T>
inline std::vector<T> flat2nd(T index, const std::vector<T> &stride) {
  std::vector<T> nd_index(stride.size());
  for (int axis = 0; axis < nd_index.size(); ++axis) {
    nd_index[axis] = index / stride[axis];
    index -= nd_index[axis] * stride[axis];
  }
  return nd_index;
}

template <typename T>
inline T inner_size(const std::vector<T> &shape, int axis) {
  using std::accumulate;
  using std::multiplies;
  axis = axis < 0 ? shape.size() + axis : axis;
  assert(0 <= axis && axis < shape.size());
  return accumulate(shape.begin() + axis, shape.end(), 1, multiplies<T>());
}

template <typename T>
inline T outer_size(const std::vector<T> &shape, int axis) {
  using std::accumulate;
  using std::multiplies;
  axis = axis < 0 ? shape.size() + axis : axis;
  assert(0 <= axis && axis < shape.size());
  return accumulate(shape.begin(), shape.begin() + axis, 1, multiplies<T>());
}

template <typename T>
inline bool increment(std::vector<T> &index, const std::vector<T> &shape) {
  assert(index.size() == shape.size());
  for (int axis = index.size() - 1; axis >= 0; axis--) {
    if (index.at(axis) + 1 < shape.at(axis)) {
      index.at(axis) += 1;
      return true;
    } else {
      index.at(axis) = 0;
    }
  }
  return false;
}

} // namespace ndi
} // namespace nbla
#endif
