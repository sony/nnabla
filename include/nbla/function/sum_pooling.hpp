// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

/** SumPooling
*/
#ifndef __NBLA_FUNCTION_SUMPOOLING_HPP__
#define __NBLA_FUNCTION_SUMPOOLING_HPP__

//#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function/utils/base_pooling.hpp>
#include <nbla/function_registry.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(SumPooling, const vector<int> &,
                              const vector<int> &, bool, const vector<int> &,
                              bool);

/** Sum pooling operator.

@copydetails BasePooling
\ingroup FunctionImplGrp
*/

template <typename T>
class SumPooling
    : public BasePooling<T, const vector<int> &, const vector<int> &, bool,
                         const vector<int> &, bool> {
public:
  SumPooling(const Context &ctx, const vector<int> &kernel,
             const vector<int> &stride, bool ignore_border,
             const vector<int> &pad, bool channel_last)
      : NBLA_THIS_TYPE::base_pooling_type(ctx, kernel, stride, ignore_border,
                                          pad, channel_last) {}

  virtual ~SumPooling() {}
  virtual shared_ptr<Function> copy() const {
    return create_SumPooling(this->ctx_, this->kernel_, this->stride_,
                             this->ignore_border_, this->pad_,
                             this->channel_last_);
  }
  virtual string name() { return "SumPooling"; }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
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
