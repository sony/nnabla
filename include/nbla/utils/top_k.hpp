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

#ifndef __NBLA_UTILS_TOPK_HPP__
#define __NBLA_UTILS_TOPK_HPP__

#include <nbla/common.hpp>

#include <algorithm>
#include <utility>

namespace nbla {

/* top_k(x, n, k, out) writes the indices of the k largest elements
 * from x[0 ... n-1] into out[0 ... k-1]. The element out[0] is the
 * index of the largest element in x. The elements out[1 ... k-1]
 * are the indices of subsequently smaller or equal elements in x.
 */

template <typename T, bool largest>
inline void top_k(const T *x, const size_t n, const size_t k, size_t *out) {
  using greater =
      typename std::conditional<largest, std::greater<T>, std::less<T>>::type;

  struct cmp {
    bool operator()(const std::pair<T, size_t> &a,
                    const std::pair<T, size_t> &b) {
      return greater()(a.first, b.first);
    }
  };

  vector<std::pair<T, size_t>> heap(k);
  for (size_t i = 0; i < k; ++i) {
    heap[i] = std::make_pair(x[i], i);
  }

  std::make_heap(heap.begin(), heap.end(), cmp());

  for (size_t i = k; i < n; ++i) {
    const auto x_at_i = x[i];
    if (greater()(x_at_i, heap[0].first)) {
      std::pop_heap(heap.begin(), heap.end(), cmp());
      heap[heap.size() - 1] = std::make_pair(x_at_i, i);
      std::push_heap(heap.begin(), heap.end(), cmp());
    }
  }
  std::sort_heap(heap.begin(), heap.end(), cmp());

  for (size_t i = 0; i < k; ++i) {
    out[i] = heap[i].second;
  }
}

/* top_k_abs(x, n, k, out) behaves identical to top_k() except that
 * it considers the absolute values of elements in x.
 */

template <typename T, bool largest>
inline void top_k_abs(const T *x, const size_t n, const size_t k, size_t *out) {
  using greater =
      typename std::conditional<largest, std::greater<T>, std::less<T>>::type;

  struct cmp {
    bool operator()(const std::pair<T, size_t> &a,
                    const std::pair<T, size_t> &b) {
      return greater()(a.first, b.first);
    }
  };

  vector<std::pair<T, size_t>> heap(k);
  for (size_t i = 0; i < k; ++i) {
    heap[i] = std::make_pair(x[i] < 0 ? -x[i] : x[i], i);
  }

  std::make_heap(heap.begin(), heap.end(), cmp());

  for (size_t i = k; i < n; ++i) {
    const auto x_at_i = x[i] < 0 ? -x[i] : x[i];
    if (greater()(x_at_i, heap[0].first)) {
      std::pop_heap(heap.begin(), heap.end(), cmp());
      heap[heap.size() - 1] = std::make_pair(x_at_i, i);
      std::push_heap(heap.begin(), heap.end(), cmp());
    }
  }
  std::sort_heap(heap.begin(), heap.end(), cmp());

  for (size_t i = 0; i < k; ++i) {
    out[i] = heap[i].second;
  }
}
} // namespace nbla
#endif
