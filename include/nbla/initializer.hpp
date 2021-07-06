// Copyright 2019,2020,2021 Sony Corporation.
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
// Include nnabla header files

/** Initializer
*/
#ifndef __NBLA_INITIALIZER_HPP__
#define __NBLA_INITIALIZER_HPP__

#include <nbla/nd_array.hpp>

namespace nbla {

NBLA_API float calc_normal_std_he_forward(int n_map_in, int n_map_out,
                                          int kernel_dim_product);

NBLA_API float calc_normal_std_he_backward(int n_map_in, int n_map_out,
                                           int kernel_dim_product);

NBLA_API float calc_normal_std_glorot(int n_map_in, int n_map_out,
                                      int kernel_dim_product);

NBLA_API float calc_uniform_lim_glorot(int n_map_in, int n_map_out,
                                       int kernel_dim_product);

class NBLA_API Initializer {
public:
  Initializer();
  virtual ~Initializer();
  virtual void initialize(NdArrayPtr parameter);
};

class NBLA_API UniformInitializer : public Initializer {
public:
  UniformInitializer();
  UniformInitializer(float lower, float upper);
  void initialize(NdArrayPtr param);

private:
  float lower_;
  float upper_;
};

class NBLA_API ConstantInitializer : public Initializer {
public:
  ConstantInitializer();
  ConstantInitializer(float value);
  void initialize(NdArrayPtr param);

private:
  float value_;
};

class NBLA_API NormalInitializer : public Initializer {
public:
  NormalInitializer();
  NormalInitializer(float mu, float sigma);
  void initialize(NdArrayPtr param);

private:
  float mu_;
  float sigma_;
};

class NBLA_API UniformIntInitializer : public Initializer {
public:
  UniformIntInitializer();
  UniformIntInitializer(int lower, int upper);
  void initialize(NdArrayPtr param);

private:
  int lower_;
  int upper_;
};
}
#endif
