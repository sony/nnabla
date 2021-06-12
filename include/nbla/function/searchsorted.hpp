// Copyright 2021 Sony Group Corporation.
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

#ifndef NBLA_FUNCTION_SEARCHSORTED_HPP
#define NBLA_FUNCTION_SEARCHSORTED_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(SearchSorted, bool);

/**
    SearchSorted searches for the index of values in the innermost dimension in
sorted_sequence such that after inserting values in the sorted_sequence it
remains sorted.

Inputs:
- sorted_sequence: N-D array
- values: N-D array

Outputs:
- N-D array with the same shape as values

@param right if true last index is considered in case of repeated elements
\ingroup FunctionImplGrp
 */
template <typename T> class SearchSorted : public BaseFunction<bool> {
protected:
  bool right_;
  size_t ss_last_dim_, v_last_dim_, inner_size_;

public:
  SearchSorted(const Context &ctx, bool right)
      : BaseFunction(ctx, right), right_(right) {}
  virtual ~SearchSorted() {}
  virtual shared_ptr<Function> copy() const {
    return create_SearchSorted(ctx_, right_);
  }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "SearchSorted"; }
  // TODO: If any of data/grad storage is inplaced with any of output, you must
  // override some of these. See doc in function.hpp.
  // virtual int inplace_data(int i) const {
  // }
  // virtual int inplace_data_with(int i) const {
  // }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }
};
}
#endif
