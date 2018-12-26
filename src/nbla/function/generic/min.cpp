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

#include <nbla/function/min.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Min, const vector<int> &, bool, bool, bool);

template <typename T>
void Min<T>::forward_impl_reduce(const T *x, T *y, int outer_size,
                                 int reduction_size) {
  // Note: Saving index is a bit inefficient if backward is not required.
  auto _cast = [this](Variable *v) {
    return v->cast_data_and_get_pointer<int>(this->ctx_, true);
  };
  int *ind = _cast(this->index_buff_.get());
  for (int o = 0; o < outer_size; ++o) {
    int mi = 0;
    T m = 1e+8;
    for (int i = 0; i < reduction_size; ++i) {
      const T v = x[o * reduction_size + i];
      if (v < m) {
        m = v;
        mi = i;
      }
    }
    y[o] = m;
    ind[o] = mi;
  }
}

} // namespace nbla
