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

#ifndef __NBLA_UTILS_AXISUTILS_HPP__
#define __NBLA_UTILS_AXISUTILS_HPP__

#include <nbla/exception.hpp>
#include <vector>

namespace nbla {

inline void refine_axis(int &axis, int ndim) {

  NBLA_CHECK(axis < ndim && axis >= -ndim, error_code::value,
             "axis must be in the range of [-ndim, ndim). axis : "
             "%d, ndim: %d.",
             axis, ndim);
  axis = axis < 0 ? axis + ndim : axis;
}

inline void refine_axes(vector<int> &axes, int ndim) {

  for (int &a : axes) {
    NBLA_CHECK(
        a < ndim && a >= -ndim, error_code::value,
        "each axis element must be in the range of [-ndim, ndim). axis : "
        "%d, ndim: %d.",
        a, ndim);

    a = (a < 0) ? ndim + a : a;
  }
}
}
#endif