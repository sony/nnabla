// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Include nnabla header files

#include <climits>
#include <random>

#include <nbla/context.hpp>
#include <nbla/initializer.hpp>
using std::make_shared;

namespace nbla {

using namespace std;
std::random_device seed_gen;
std::default_random_engine engine(seed_gen());
std::uniform_real_distribution<> uniform_real(0.0, 1.0);
std::normal_distribution<> normal(0.0, 1.0);
std::uniform_int_distribution<> uniform_int(0, INT_MAX);

nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

float calc_normal_std_he_forward(int n_map_in, int n_map_out,
                                 int kernel_dim_product) {
  return sqrt(2. / (kernel_dim_product * n_map_in));
}

float calc_normal_std_he_backward(int n_map_in, int n_map_out,
                                  int kernel_dim_product) {
  return sqrt(2. / (kernel_dim_product * n_map_out));
}

float calc_normal_std_glorot(int n_map_in, int n_map_out,
                             int kernel_dim_product) {
  return sqrt(2. / (kernel_dim_product * n_map_in + n_map_out));
}

float calc_uniform_lim_glorot(int n_map_in, int n_map_out,
                              int kernel_dim_product) {
  return sqrt(6. / (kernel_dim_product * n_map_in + n_map_out));
}

Initializer::Initializer() {}
Initializer::~Initializer() {}
void Initializer::initialize(NdArrayPtr parameter) {}

UniformInitializer::UniformInitializer()
    : Initializer(), lower_(-1.0), upper_(1.0) {}
UniformInitializer::UniformInitializer(float lower, float upper)
    : Initializer(), lower_(lower), upper_(upper) {}
void UniformInitializer::initialize(NdArrayPtr param) {
  const int size = param->size();
  Array *arr = param->cast(get_dtype<float_t>(), cpu_ctx, false);
  float_t *param_d = arr->pointer<float_t>();
  for (int i = 0; i < size; i++)
    param_d[i] = (upper_ - lower_) * uniform_real(engine) + lower_;
}

ConstantInitializer::ConstantInitializer() : Initializer(), value_(0.0) {}
ConstantInitializer::ConstantInitializer(float value)
    : Initializer(), value_(value) {}
void ConstantInitializer::initialize(NdArrayPtr param) {
  const int size = param->size();
  Array *arr = param->cast(get_dtype<float_t>(), cpu_ctx, false);
  float_t *param_d = arr->pointer<float_t>();
  for (int i = 0; i < size; i++)
    param_d[i] = value_;
}

NormalInitializer::NormalInitializer() : Initializer(), mu_(0.0), sigma_(1.0) {}
NormalInitializer::NormalInitializer(float mu, float sigma)
    : Initializer(), mu_(mu), sigma_(sigma) {}
void NormalInitializer::initialize(NdArrayPtr param) {
  const int size = param->size();
  Array *arr = param->cast(get_dtype<float_t>(), cpu_ctx, false);
  float_t *param_d = arr->pointer<float_t>();
  for (int i = 0; i < size; i++)
    param_d[i] = mu_ + sigma_ * normal(engine);
}

UniformIntInitializer::UniformIntInitializer()
    : Initializer(), lower_(0), upper_(INT_MAX) {}
UniformIntInitializer::UniformIntInitializer(int lower, int upper)
    : Initializer(), lower_(lower), upper_(upper) {}
void UniformIntInitializer::initialize(NdArrayPtr param) {
  const int size = param->size();
  Array *arr = param->cast(get_dtype<int>(), cpu_ctx, false);
  int *param_d = arr->pointer<int>();
  for (int i = 0; i < size; i++)
    param_d[i] = uniform_int(engine) % (upper_ - lower_) + lower_;
}
}
