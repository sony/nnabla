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

#ifndef __NBLA_UTILS_ND_INDEX_HPP__
#define __NBLA_UTILS_ND_INDEX_HPP__

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
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

// Calculate strides from `shape` and return as a new allocated vector.
template <typename T>
inline std::vector<T> strides(const std::vector<T> &shape) {
  std::vector<T> v(shape.size(), 1);
  std::copy(shape.begin() + 1, shape.end(), v.begin());
  std::partial_sum(v.rbegin(), v.rend(), v.rbegin(), std::multiplies<T>());
  return v;
}

// Compute the flat offset for the multidimensional `index` within the shape
// determined by `strides`.
template <typename T>
inline T nd2flat(const std::vector<T> &index, const std::vector<T> &stride) {
  assert(index.size() <= stride.size());
  return std::inner_product(index.begin(), index.end(), stride.begin(), 0);
}

// Compute the flat offset for the multidimensional `index` within the shape
// determined by `strides` restricted to the outer dimensions until and not
// including `axis`.
template <typename T>
inline T nd2flat(const std::vector<T> &index, const std::vector<T> &stride,
                 int axis) {
  using std::inner_product;
  assert(stride.size() <= std::numeric_limits<int>::max());
  assert(index.size() <= stride.size());
  if (axis < 0)
    axis += static_cast<int>(stride.size());
  assert(0 <= axis && axis < static_cast<int>(index.size()));
  return inner_product(index.begin(), index.begin() + axis, stride.begin(), 0);
}

// Compute the flat offset for the multidimensional `index` within the shape
// determined by `strides` restricted to the dimensions from and including
// `axis.first` until and excluding `axis.second`.
template <typename T>
inline T nd2flat(const std::vector<T> &index, const std::vector<T> &stride,
                 std::pair<int, int> axis) {
  assert(stride.size() <= std::numeric_limits<int>::max());
  assert(index.size() <= stride.size());
  if (axis.first < 0)
    axis.first += static_cast<int>(stride.size());
  if (axis.second < 0)
    axis.second += static_cast<int>(stride.size());
  assert(0 <= axis.first && axis.first <= axis.second);
  assert(axis.second <= static_cast<int>(index.size()));
  T result = 0;
  for (; axis.first < axis.second; axis.first++) {
    result += index[axis.first] * stride[axis.first];
  }
  return result;
}

// Convert a flat `index` to a multidimensional index according to `stride`.
template <typename T>
inline std::vector<T> flat2nd(T index, const std::vector<T> &stride) {
  assert(stride.size() <= std::numeric_limits<int>::max());
  std::vector<T> nd_index(stride.size());
  for (int axis = 0; axis < static_cast<int>(nd_index.size()); ++axis) {
    nd_index[axis] = index / stride[axis];
    index -= nd_index[axis] * stride[axis];
  }
  return nd_index;
}

// Calculate the size of the subspace of `shape` that goes from axis to last.
//
// std::vector<int> shape = {2, 3, 4, 5, 6};
// ndi::inner_size(shape, 3) => 30
//
template <typename T>
inline T inner_size(const std::vector<T> &shape, int axis) {
  using std::accumulate;
  using std::multiplies;
  assert(shape.size() <= std::numeric_limits<int>::max());
  if (axis < 0)
    axis += static_cast<int>(shape.size());
  assert(0 <= axis && axis < static_cast<int>(shape.size()));
  return accumulate(shape.begin() + axis, shape.end(), 1, multiplies<T>());
}

// Calculate the size of the subspace of `shape` that goes from 0 to axis - 1.
//
// std::vector<int> shape = {2, 3, 4, 5, 6};
// ndi::outer_size(shape, 3) => 24
//
template <typename T>
inline T outer_size(const std::vector<T> &shape, int axis) {
  using std::accumulate;
  using std::multiplies;
  assert(shape.size() <= std::numeric_limits<int>::max());
  if (axis < 0)
    axis += static_cast<int>(shape.size());
  assert(0 <= axis && axis < static_cast<int>(shape.size()));
  return accumulate(shape.begin(), shape.begin() + axis, 1, multiplies<T>());
}

// Increment by one the `index` vector within the given `shape`, wrapping over
// axes as needed. The return value is true for all increments except when all
// axes wrap to zero at once. Example:
//
// std::vector<int> index = {0, 0}, shape = {2, 2};
// do {
//   std::cout << ndi::str(index) << " ";
// } while (ndi::increment(index, shape));
//
// produces "[0, 0] [0 1] [1, 0] [1, 1]"
//
template <typename T>
inline bool increment(std::vector<T> &index, const std::vector<T> &shape) {
  assert(shape.size() <= std::numeric_limits<int>::max());
  assert(index.size() == shape.size());
  for (int axis = static_cast<int>(index.size()) - 1; axis >= 0; axis--) {
    if (index[axis] + 1 < shape[axis]) {
      index[axis] += 1;
      return true;
    } else {
      index[axis] = 0;
    }
  }
  return false;
}

// Increment by one the `index` vector within the subspace of `shape` that
// ranges from 0 to `axis`. The return value is true for all increments except
// when all axes of the subspace of `shape` wrap to zero at once. Example:
//
// std::vector<int> index = {0, 0, 0}, shape = {2, 2, 2};
// do {
//   std::cout << ndi::str(index) << " ";
// } while (ndi::increment(index, shape, 1));
//
// produces "[0, 0, 0] [0, 1, 0] [1, 0, 0] [1, 1, 0]"
//
template <typename T>
inline bool increment(std::vector<T> &index, const std::vector<T> &shape,
                      int axis) {
  assert(shape.size() <= std::numeric_limits<int>::max());
  assert(index.size() == shape.size());
  if (axis < 0)
    axis += static_cast<int>(index.size());
  assert(0 <= axis && axis < static_cast<int>(index.size()));
  for (; axis >= 0; axis--) {
    if (index[axis] + 1 < shape[axis]) {
      index[axis] += 1;
      return true;
    } else {
      index[axis] = 0;
    }
  }
  return false;
}

// Multiply two vectors of potentially different sizes to a new vector that has
// the size of the larger input vector. The missing elements from the shorter
// input vector are interpreted as 1.
template <typename Ta, typename Tb>
inline std::vector<Ta> multiply(const std::vector<Ta> &a,
                                const std::vector<Tb> &b) {
  std::vector<Ta> r(std::max(a.size(), b.size()), 1);
  if (a.size() < b.size()) {
    std::copy_backward(a.begin(), a.end(), r.end());
    for (typename std::vector<Ta>::size_type i = 0; i < r.size(); i++)
      r.at(i) *= b.at(i);
  } else {
    std::copy_backward(b.begin(), b.end(), r.end());
    for (typename std::vector<Ta>::size_type i = 0; i < r.size(); i++)
      r.at(i) *= a.at(i);
  }
  return r;
}

// Return a copy of vector `a` expanded to size `n` by prepending n - a.size()
// elements with value `v`. If `a` is intended as a multidimensional index or a
// shape then this function is expanding into outer dimensions.
template <typename T>
inline std::vector<T> expand(const std::vector<T> &a, size_t n, const T &v) {
  std::vector<T> r(std::max(a.size(), n), v);
  std::copy_backward(a.begin(), a.end(), r.end());
  return r;
}

template <typename T>
inline std::vector<T> make_index(size_t ndim, const T &init = T()) {
  return std::vector<T>(ndim, init);
}

template <typename T>
inline std::vector<T> batch_reduced_shape(std::vector<T> &shape,
                                          int reduce_axis_up_to) {
  assert(reduce_axis_up_to < shape.size());
  std::vector<T> ret(shape.size() - reduce_axis_up_to + 1);
  ret[0] = outer_size(shape, reduce_axis_up_to);
  std::copy(shape.cbegin() + reduce_axis_up_to, shape.cend(), ret.begin() + 1);
  return ret;
}

} // namespace ndi
} // namespace nbla
#endif
