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

/** AveragePooling
*/
#ifndef __NBLA_FUNCTION_AVERAGEPOOLING_HPP__
#define __NBLA_FUNCTION_AVERAGEPOOLING_HPP__

#include <nbla/function.hpp>
#include <nbla/function/utils/base_pooling.hpp>
#include <nbla/function_registry.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(AveragePooling, const vector<int> &,
                              const vector<int> &, bool, const vector<int> &,
                              bool, bool);

/** Average pooling operator.

@copydetails BasePooling
@param including_pad If true, kernel size will be used for denominator of
                     averaging even when the kernel overlaps border, otherwise
                     the count of cells inside input tensor will be used.
\ingroup FunctionImplGrp
*/

template <typename T>
class AveragePooling
    : public BasePooling<T, const vector<int> &, const vector<int> &, bool,
                         const vector<int> &, bool, bool> {
protected:
  bool including_pad_;

public:
  AveragePooling(const Context &ctx, const vector<int> &kernel,
                 const vector<int> &stride, bool ignore_border,
                 const vector<int> &pad, bool channel_last, bool including_pad)
      : NBLA_THIS_TYPE::base_pooling_type(ctx, kernel, stride, ignore_border,
                                          pad, channel_last, including_pad),
        including_pad_(including_pad) {}

  virtual ~AveragePooling() {}
  virtual shared_ptr<Function> copy() const {
    return create_AveragePooling(this->ctx_, this->kernel_, this->stride_,
                                 this->ignore_border_, this->pad_,
                                 this->channel_last_, including_pad_);
  }
  virtual string name() { return "AveragePooling"; }

protected:
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
}
#endif
