// Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

#include <assert.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>

#include <nbla/computation_graph/variable.hpp>
#include <nbla/initializer.hpp>
#include <nbla/logger.hpp>
#include <nbla/parametric_functions.hpp>

// Protobuf
#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "nbla_utils/parameters.hpp"
#include "nnabla.pb.h"

namespace nbla {
namespace utils {

const nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

class InitializerProtoParam : public ::nbla::Initializer {
  const Parameter &param_;

public:
  InitializerProtoParam(const Parameter &param) : param_(param) {}
  virtual ~InitializerProtoParam() {}
  virtual void initialize(NdArrayPtr parameter) {
    Array *arr = parameter->cast(get_dtype<float_t>(), cpu_ctx, false);
    float_t *param_d = arr->pointer<float_t>();
    auto p_data = param_.data();
    NBLA_CHECK(p_data.size() == parameter->size(), error_code::value,
               "Inconsistent size in proto parameter %s (%d != %d)",
               param_.variable_name().c_str(), (int)p_data.size(),
               (int)parameter->size());
    for (int i = 0; i < p_data.size(); i++) {
      param_d[i] = p_data[i];
    }
  }
};

// ----------------------------------------------------------------------
// Load parameters from file
// ----------------------------------------------------------------------
void load_parameters(ParameterDirectory &pd, string filename) {
  NNablaProtoBuf proto;
  std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    NBLA_LOG_WARN("Error in opening file {}", filename);
    return;
  }
  proto.ParseFromIstream(&ifs);
  for (auto it = proto.parameter().begin(); it != proto.parameter().end();
       it++) {
    const string &name = it->variable_name();
    Shape_t shape(it->shape().dim().begin(), it->shape().dim().end());
    bool need_grad = it->need_grad();
    InitializerProtoParam init_param(*it);
    pd.get_parameter_or_create(name, shape, &init_param, need_grad);
  }
  NBLA_LOG_INFO("Load parameters from {}", filename);
}

// ----------------------------------------------------------------------
// Save parameters to file
// ----------------------------------------------------------------------
void save_parameters(ParameterDirectory &pd, string filename) {
  std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
  if (!ofs.is_open()) {
    NBLA_LOG_WARN("Error in opening file {}", filename);
    return;
  }

  NNablaProtoBuf params;
  auto pd_param = pd.get_parameters();
  for (auto it = pd_param.begin(); it != pd_param.end(); it++) {

    string name = it->first;
    VariablePtr variable = it->second;

    Parameter *parameter = params.add_parameter();
    parameter->set_variable_name(name);
    CgVariablePtr cg_var = pd.get_parameter(name);
    parameter->set_need_grad(cg_var->need_grad());

    float *data = variable->template cast_data_and_get_pointer<float>(cpu_ctx);
    for (int i = 0; i < variable->size(); i++)
      parameter->add_data(data[i]);

    Shape *shape = parameter->mutable_shape();
    for (int i = 0; i < variable->shape().size(); i++)
      shape->add_dim(variable->shape()[i]);
  }
  params.SerializeToOstream(&ofs);
  NBLA_LOG_INFO("Saved parameters to {}", filename);
}
}
}
