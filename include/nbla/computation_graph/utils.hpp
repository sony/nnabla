// Copyright 2020,2021 Sony Corporation.
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

#ifndef __NBLA_UTILS_CG_UTILS_HPP__
#define __NBLA_UTILS_CG_UTILS_HPP__

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/function/reshape.hpp>
#include <nbla/function/slice.hpp>
#include <nbla/variable.hpp>

namespace nbla {
namespace cg_utils {

inline shared_ptr<CgVariable> get_item_nd(const Context &ctx,
                                          shared_ptr<CgVariable> in,
                                          vector<int> slice_shape) {

  vector<CgVariablePtr> out_shape;
  if (slice_shape.size() > 2) {
    NBLA_ERROR(error_code::not_implemented,
               "get_item_nd does not support sliceing more than 2 dims.")
  }
  vector<int> start(in->variable()->ndim(), 0);
  vector<int> step(in->variable()->ndim(), 1);
  Shape_t in_shape = in->variable()->shape();
  vector<int> stop{in_shape.begin(), in_shape.end()};

  for (vector<int>::size_type i = 0; i < slice_shape.size(); i++) {
    start[i] += slice_shape[i];
  }

  for (vector<int>::size_type i = 0; i < slice_shape.size(); i++) {
    if (start[i] + 1 < stop[i])
      stop[i] = start[i] + 1;
  }

  auto slice = make_shared<CgFunction>(create_Slice(ctx, start, stop, step));
  out_shape = connect(slice, {in}, 1);

  vector<int> shape_out{in_shape.begin() + slice_shape.size(), in_shape.end()};
  auto reshape = make_shared<CgFunction>(create_Reshape(ctx, shape_out, true));
  out_shape = connect(reshape, out_shape, 1);

  return out_shape[0];
}

template <typename T>
inline void copy_data_cgvariable_to_variable(const Context &ctx,
                                             shared_ptr<CgVariable> in,
                                             Variable *out) {
  auto var_in = in->variable();
  auto *data_in = var_in->get_data_pointer<T>(ctx);
  auto out_ptr = out->cast_data_and_get_pointer<T>(ctx, true);
  for (int j = 0; j < var_in.get()->size(); j++) {
    out_ptr[j] = data_in[j];
  }
}

} // namespace cg_utils
} // namespace nbla
#endif
