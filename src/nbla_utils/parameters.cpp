// Copyright 2019,2020,2021 Sony Corporation.
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
#include <nbla/std.hpp>

// Protobuf
#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "hdf5_wrapper.hpp"
#include "nbla_utils/parameters.hpp"
#include "nnabla.pb.h"
#include "parameters_impl.hpp"

using namespace std;

namespace nbla {
namespace utils {
namespace { // Internal functions and variables

const nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
const int MAX_NAME = 1024;

string get_extension(const string &filename) {
  int ep = filename.find_last_of(".");
  if (ep < 0) {
    NBLA_ERROR(error_code::value, "File: %s has no extension name.",
               filename.c_str());
  }
  string ext = filename.substr(ep, filename.size() - ep);
  return ext;
}

void load_parameters_from_proto(NNablaProtoBuf &param, ParameterVector &pv) {
  for (auto it = param.parameter().begin(); it != param.parameter().end();
       it++) {
    const string &name = it->variable_name();
    Shape_t shape(it->shape().dim().begin(), it->shape().dim().end());
    bool need_grad = it->need_grad();
    CgVariablePtr cg_v = make_shared<CgVariable>(shape, need_grad);
    float *data =
        cg_v->variable()->template cast_data_and_get_pointer<float>(cpu_ctx);
    auto &p_data = it->data();
    NBLA_CHECK(p_data.size() == cg_v->variable()->size(), error_code::value,
               "Inconsistent size in proto parameter %s (%d != %d)",
               name.c_str(), (int)p_data.size(), (int)cg_v->variable()->size());
    for (int i = 0; i < p_data.size(); i++) {
      data[i] = p_data[i];
    }
    pv.push_back({name, cg_v});
  }
}

} // namespace

// ----------------------------------------------------------------------
// load parameters from protobuf file's buffer
// ----------------------------------------------------------------------
bool load_parameters_pb(ParameterVector &pv, char *buffer, int size) {
  constexpr int size_1024_mb = 1024 << 20;
#if GOOGLE_PROTOBUF_VERSION < 3006001
  constexpr int size_128_mb = 128 << 20;
#endif
  NNablaProtoBuf param;
  google::protobuf::io::ArrayInputStream input(buffer, size);
  google::protobuf::io::CodedInputStream coded_input(&input);
#if GOOGLE_PROTOBUF_VERSION < 3006001
  coded_input.SetTotalBytesLimit(size_1024_mb, size_128_mb);
#else
  coded_input.SetTotalBytesLimit(size_1024_mb);
#endif
  bool ret = param.ParseFromCodedStream(&coded_input);
  if (!ret) {
    NBLA_ERROR(error_code::value, "Cannot load parameters from buffer!");
    return false;
  }

  load_parameters_from_proto(param, pv);
  return true;
}

// ----------------------------------------------------------------------
// load parameters from protobuf file
// ----------------------------------------------------------------------
bool load_parameters_pb(ParameterVector &pv, string filename) {
  NNablaProtoBuf param;
  std::ifstream ifs(filename.c_str(), std::ios::binary);
  if (!ifs.is_open()) {
    NBLA_LOG_WARN("Error in opening file {}", filename);
    return false;
  }
  param.ParseFromIstream(&ifs);

  load_parameters_from_proto(param, pv);
  return true;
}

// ----------------------------------------------------------------------
// load parameters from file which is specified by filename
// choose the format by its file extension name.
// ----------------------------------------------------------------------
bool load_parameters(ParameterVector &pv, string filename) {
  bool ret = false;
  string ext = get_extension(filename);
  if (ext == ".h5") {
    ret = load_parameters_h5(pv, filename);
  } else if (ext == ".protobuf") {
    ret = load_parameters_pb(pv, filename);
  } else {
    NBLA_ERROR(error_code::value, "Not supported file extension: %s",
               filename.c_str());
  }
  return ret;
}

// ----------------------------------------------------------------------
// Load parameters from file
// ----------------------------------------------------------------------
bool load_parameters(ParameterDirectory &pd, string filename) {
  ParameterVector pv;
  bool ret = load_parameters(pv, filename);

  if (!ret) {
    NBLA_LOG_INFO("Cannot load parameter file: %s\n", filename.c_str());
    return false;
  }

  for (auto it = pv.begin(); it != pv.end(); ++it) {
    pd.get_parameter_or_create(it->first, it->second);
  }
  NBLA_LOG_INFO("Load parameters from {}", filename);

  return true;
}

// ----------------------------------------------------------------------
// Load parameters from h5 file buffer
// ----------------------------------------------------------------------
bool load_parameters_h5(ParameterDirectory &pd, char *buf, int size) {
  ParameterVector pv;
  bool ret = load_parameters_h5(pv, buf, size);

  if (!ret) {
    NBLA_LOG_INFO("Cannot load parameter file buffer!\n");
    return false;
  }

  for (auto it = pv.begin(); it != pv.end(); ++it) {
    pd.get_parameter_or_create(it->first, it->second);
  }

  return true;
}

// ----------------------------------------------------------------------
// Load parameters from pb file buffer
// ----------------------------------------------------------------------
bool load_parameters_pb(ParameterDirectory &pd, char *buf, int size) {
  ParameterVector pv;
  bool ret = load_parameters_pb(pv, buf, size);

  if (!ret) {
    NBLA_LOG_INFO("Cannot load parameter file buffer!\n");
    return false;
  }

  for (auto it = pv.begin(); it != pv.end(); ++it) {
    pd.get_parameter_or_create(it->first, it->second);
  }

  return true;
}

// ----------------------------------------------------------------------
// save parameters as pb to a stream output
// ----------------------------------------------------------------------
void save_parameters_pb_to_stream(const ParameterVector &pv, ostream *ofs) {
  NNablaProtoBuf params;
  for (auto it = pv.begin(); it != pv.end(); it++) {

    string name = it->first;
    VariablePtr variable = it->second->variable();
    Parameter *parameter = params.add_parameter();
    parameter->set_variable_name(name);
    CgVariablePtr cg_var = it->second;
    parameter->set_need_grad(cg_var->need_grad());

    float *data = variable->template cast_data_and_get_pointer<float>(cpu_ctx);
    for (int i = 0; i < variable->size(); i++)
      parameter->add_data(data[i]);

    Shape *shape = parameter->mutable_shape();
    for (Shape_t::size_type i = 0; i < variable->shape().size(); i++)
      shape->add_dim(variable->shape()[i]);
  }
  params.SerializeToOstream(ofs);
}

// ----------------------------------------------------------------------
// save parameters to a .protobuf file specified by filename
// ----------------------------------------------------------------------
bool save_parameters_pb(const ParameterVector &pv, string filename) {
  std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
  if (!ofs.is_open()) {
    NBLA_LOG_WARN("Error in opening file {}", filename);
    return false;
  }

  save_parameters_pb_to_stream(pv, &ofs);
  NBLA_LOG_INFO("Saved parameters to {}", filename);
  return true;
}

// ----------------------------------------------------------------------
// save parameters to a .protobuf file specified by filename
// ----------------------------------------------------------------------
bool save_parameters_pb(const ParameterVector &pv, char *buffer,
                        unsigned int &size) {
  std::stringstream ss;

  save_parameters_pb_to_stream(pv, &ss);
  ss.seekp(0, std::ios::end);
  if (buffer == NULL) {
    size = ss.tellp();
  } else {
    if (size < ss.tellp()) {
      NBLA_ERROR(error_code::memory,
                 "Required buffer size %d is not satisfied by %d", ss.tellp(),
                 size)
    }
    size = ss.tellp();
    ss.seekp(0, std::ios::beg);
    ss.read(buffer, size);
  }

  NBLA_LOG_INFO("Saved parameters to buffer!");
  return true;
}

// ----------------------------------------------------------------------
// save parameters to a file specified by filename
// ----------------------------------------------------------------------
bool save_parameters(const ParameterVector &pv, string filename) {
  bool ret = false;
  string ext = get_extension(filename);
  if (ext == ".h5") {
    ret = save_parameters_h5(pv, filename);
  } else if (ext == ".protobuf") {
    ret = save_parameters_pb(pv, filename);
  } else {
    NBLA_ERROR(error_code::value, "Not supported file extension: %s",
               filename.c_str());
  }
  return ret;
}

// ----------------------------------------------------------------------
// Save parameters to file
// ----------------------------------------------------------------------
bool save_parameters(ParameterDirectory &pd, string filename) {
  ParameterVector pv;
  bool ret = false;
  auto pd_param = pd.get_parameters();

  for (auto it = pd_param.begin(); it != pd_param.end(); ++it) {
    pv.push_back({it->first, pd.get_parameter(it->first)});
  }

  ret = save_parameters(pv, filename);
  if (!ret) {
    NBLA_ERROR(error_code::value, "Cannot save parameter file:%s",
               filename.c_str());
    return false;
  }
  NBLA_LOG_INFO("Saved parameters to {}", filename);
  return true;
}

// ----------------------------------------------------------------------
// Save parameters from h5 file buffer
// ----------------------------------------------------------------------
bool save_parameters_h5(ParameterDirectory &pd, char *buf, unsigned int &size) {
  ParameterVector pv;
  auto pd_param = pd.get_parameters();

  for (auto it = pd_param.begin(); it != pd_param.end(); ++it) {
    pv.push_back({it->first, pd.get_parameter(it->first)});
  }

  return save_parameters_h5(pv, buf, size);
}

// ----------------------------------------------------------------------
// Save parameters from pb file buffer
// ----------------------------------------------------------------------
bool save_parameters_pb(ParameterDirectory &pd, char *buf, unsigned int &size) {
  ParameterVector pv;
  auto pd_param = pd.get_parameters();

  for (auto it = pd_param.begin(); it != pd_param.end(); ++it) {
    pv.push_back({it->first, pd.get_parameter(it->first)});
  }

  return save_parameters_pb(pv, buf, size);
}
} // namespace utils
} // namespace nbla
