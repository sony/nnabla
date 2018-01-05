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

/** Mean
 */
#include <nbla/function/mean.hpp>
#include <nbla/utils/eigen.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Mean, const vector<int> &, bool);

template <typename T>
void Mean<T>::forward_impl_reduce(const T *x, T *y, int outer_size,
                                  int reduction_size) {
  using namespace ::nbla::eigen;
  ConstMatrixMap<T> mx(x, outer_size, reduction_size);
  ColVectorMap<T> my(y, outer_size);
  my = mx.rowwise().sum() / reduction_size;
}

template <typename T>
void Mean<T>::backward_impl_reduce(const T *dy, T *dx, int outer_size,
                                   int reduction_size, bool accum) {
  using namespace ::nbla::eigen;
  ConstColVectorMap<T> mdy(dy, outer_size);
  MatrixMap<T> mdx(dx, outer_size, reduction_size);
  if (accum)
    mdx.colwise() += mdy / reduction_size;
  else
    mdx.colwise() = mdy / reduction_size;
}

} // namespace nbla
