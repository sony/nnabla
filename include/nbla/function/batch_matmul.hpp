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

/** BatchMatmul
 */
#ifndef __NBLA_FUNCTION_BATCHMATMUL_HPP__
#define __NBLA_FUNCTION_BATCHMATMUL_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function/broadcast.hpp>
#include <nbla/function_registry.hpp>

#include <functional>
#include <numeric>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(BatchMatmul, bool, bool);

/**Batch matrix multiplication.

Two of batches of matrices are multiplied for each sample in a batch. A batch of
matrices is composed as [..., P, Q] where the last two dimensions compose matrix
dimensions, and the first dimensions up to the third last dimension are
considered as batch samples.

Inputs:
- a: N-D array with >= 2-dim. The last two dimensions will be treated as
     a matrix.
- b: N-D array with >= 2-dim. The last two dimensions will be treated as
     a matrix. The product of the size of 0-th dimension through the size of the
     third last dimension must be same as that of the input a.

Output:
- y: Output of sample-wise matrix multiplication in a batch.
     When a is of a shape of [N, P, Q], b is of a shape of [N, Q, R], and
     transpose options are all False, the output will be a shape of [N, P, R].

@tparam T Data type for computation.
@param transpose_a Transpose the last two axes of a in matrix multiplication.
@param transpose_b Transpose the last two axes of b in matrix multiplication.

\ingroup FunctionImplGrp
 */
template <typename T> class BatchMatmul : public BaseFunction<bool, bool> {
protected:
  bool transpose_a_;
  bool transpose_b_;
  int samples_;
  int col_a_;
  int row_a_;
  int col_b_;
  int row_b_;
  int col_y_;
  int row_y_;
  int offset_a_;
  int offset_b_;
  int offset_y_;
  FunctionPtr f_broadcast_a_;
  FunctionPtr f_broadcast_b_;

public:
  BatchMatmul(const Context &ctx, bool transpose_a, bool transpose_b)
      : BaseFunction<bool, bool>(ctx, transpose_a, transpose_b),
        transpose_a_(transpose_a), transpose_b_(transpose_b) {}
  virtual ~BatchMatmul() {}
  virtual shared_ptr<Function> copy() const {
    return create_BatchMatmul(ctx_, transpose_a_, transpose_b_);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "BatchMatmul"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
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
    if (i == 0 && j == 1) {
      return true;
    }
    if (i == 1 && j == 0) {
      return true;
    }
    return false;
  }
};
}
#endif
