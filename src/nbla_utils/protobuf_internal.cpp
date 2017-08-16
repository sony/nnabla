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

#include "protobuf_internal.hpp"
#include <nbla/computation_graph/computation_graph.hpp>

#include <fstream>
#include <map>
#include <nbla/logger.hpp>

namespace nbla_utils {
namespace NNP {

bool _proto_internal::parse_hdf5_dataset(std::string name, hid_t did) {
  hid_t sp = H5Dget_space(did);
  int rank = H5Sget_simple_extent_ndims(sp);
  hsize_t dims[rank];
  herr_t err = H5Sget_simple_extent_dims(sp, dims, NULL);
  hid_t tid = H5Dget_type(did);
  H5T_class_t t_class = H5Tget_class(tid);

  hsize_t size = H5Dget_storage_size(did);
  std::string variable_name = name.substr(1, name.length());

  NBLA_LOG_INFO("Dataset Name:[{}] type: {} size: {}", variable_name, t_class,
                size);

  float *buffer = new float[size / sizeof(float)];
  assert(buffer);
  err = H5Dread(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  if (err >= 0) {
    Parameter *param = _nbla->add_parameter();
    Shape *shape = param->mutable_shape();
    for (int i = 0; i < rank; i++) {
      shape->add_dim(dims[i]);
    }
    param->set_variable_name(variable_name);
    // WHOAMI("%s\n", variable_name.c_str());

    for (int i = 0; i < size / sizeof(float); i++) {
      param->add_data(buffer[i]);
    }
    return true;
  }
  return false;
}

bool _proto_internal::parse_hdf5_group(hid_t gid) {
  ssize_t len;
  hsize_t num = 0;
  herr_t err = H5Gget_num_objs(gid, &num);
  if (err >= 0) {
    char group_name[MAX_NAME];
    len = H5Iget_name(gid, group_name, MAX_NAME);
    if (len >= 0) {
      for (int i = 0; i < num; i++) {
        char name[MAX_NAME];
        len = H5Gget_objname_by_idx(gid, (hsize_t)i, name, (size_t)MAX_NAME);
        if (len < 0) {
          return false;
        }

        int type = H5Gget_objtype_by_idx(gid, i);
        switch (type) {
        case H5G_GROUP: {
          hid_t grpid = H5Gopen(gid, name, H5P_DEFAULT);
          parse_hdf5_group(grpid);
          H5Gclose(grpid);
          break;
        }
        case H5G_DATASET: {
          hid_t did = H5Dopen(gid, name, H5P_DEFAULT);
          std::string dataset_name(group_name);
          dataset_name += "/" + std::string(name);
          parse_hdf5_dataset(dataset_name, did);
          H5Dclose(did);
          break;
        }
        case H5G_TYPE:
          NBLA_LOG_INFO("H5G_TYPE");
          break;
        case H5G_LINK:
          NBLA_LOG_INFO("H5G_LINK");
          break;
        default:
          // TODO: Unsupported member.
          NBLA_LOG_INFO("default");
          break;
        }
      }
      return true;
    }
  }
  return false;
}

shared_ptr<nbla::CgVariable>
_proto_internal::create_cgvariable(const Network &net, std::string name) {
  Variable v = search_variable(net, name);
  assert(v.name() == "");
  return create_cgvariable(v);
}

shared_ptr<nbla::CgVariable>
_proto_internal::create_cgvariable(const Variable &var) {
  nbla::Shape_t shape;
  for (int i = 0; i < var.shape().dim_size(); i++) {
    shape.push_back(var.shape().dim(i));
  }
  if (shape[0] == -1) {
    shape[0] = _batch_size;
  }
  return std::make_shared<nbla::CgVariable>(shape, true);
}

Network _proto_internal::expand_network(const Network &orig) {
  Network net;
  net.set_name(orig.name());
  net.set_batch_size(orig.batch_size());

  // get max nest depth.
  int max_nest_depth = -1;
  for (int i = 0; i < orig.function_size(); i++) {
    int depth = orig.function(i).repeat_id_size();
    if (depth > max_nest_depth) {
      max_nest_depth = depth;
    }
  }

  if (max_nest_depth > 0) {
    // WHOAMI("max_nest_depth: %d\n", max_nest_depth);
    std::vector<int> indexes(max_nest_depth);
    std::map<std::string, std::vector<std::string>> repeat_function_queue;

    for (int i = 0; i < orig.repeat_info_size(); i++) {
      RepeatInfo *rep = net.add_repeat_info();
      rep->CopyFrom(orig.repeat_info(i));
      // WHOAMI("%d %s\n", i, rep->id().c_str());
    }

    for (int i = 0; i < orig.variable_size(); i++) {
      Variable *var = net.add_variable();
      var->CopyFrom(orig.variable(i));
      // WHOAMI("%s\n", (var->name()).c_str());
    }

    for (int i = 0; i < orig.function_size(); i++) {
      const Function &orig_func = orig.function(i);
      std::string type = orig_func.type();
      // WHOAMI("%s\n", type.c_str());

      if (type == "RepeatStart") {
      } else if (type == "RepeatStart") {
      } else if (type == "RepeatEnd") {
      } else if (type == "RecurrentInput") {
      } else if (type == "RecurrentOutput") {
      } else if (type == "Delay") {
      }

      if (orig_func.repeat_id_size() == 0) {
        Function *func = net.add_function();
        func->CopyFrom(orig_func);
      } else {
        // WHOAMI("%d %s\n", i, orig_func.name().c_str());
        for (int j = 0; j < orig_func.repeat_id_size(); j++) {
          // WHOAMI("%d %d %s\n", i, j, orig_func.repeat_id(j).c_str());
        }
      }
    }
  } else {
    // WHOAMI("\n");
    net.CopyFrom(orig);
  }

  return net;
}

const Network &_proto_internal::search_network(std::string name) {
  NBLA_LOG_INFO("    Searching net {}", name);
  for (int i = 0; i < _nbla->network_size(); i++) {
    if (_nbla->network(i).name() == name) {
      NBLA_LOG_INFO("      Found at {}", i);

      if (_batch_size < 0) {
        _batch_size = _nbla->network(i).batch_size();
        NBLA_LOG_INFO("      Using default batch size {}", _batch_size);
      }

      return _nbla->network(i);
    }
  }
  static const Network null_net;
  return null_net;
}

const Parameter &_proto_internal::search_parameter(std::string name) {
  NBLA_LOG_INFO("    Searching param {}", name);
  for (int i = 0; i < _nbla->parameter_size(); i++) {
    if (_nbla->parameter(i).variable_name() == name) {
      NBLA_LOG_INFO("      Found at {}", i);
      return _nbla->parameter(i);
    }
  }
  static const Parameter null_net;
  return null_net;
}

const Variable &_proto_internal::search_variable(const Network &network,
                                                 std::string name) {
  NBLA_LOG_INFO("    Searching variable {}", name);
  for (int i = 0; i < network.variable_size(); i++) {
    if (network.variable(i).name() == name) {
      NBLA_LOG_INFO("      Found at {}", i);
      return network.variable(i);
    }
  }
  static const Variable null_var;
  return null_var;
}

bool _proto_internal::add_prototxt(std::string filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  google::protobuf::io::ZeroCopyInputStream *input =
      new google::protobuf::io::FileInputStream(fd);
  google::protobuf::TextFormat::Merge(input, _nbla);
  delete input;
  close(fd);
  return true;
}

bool _proto_internal::add_prototxt(char *buffer, int size) {
  google::protobuf::io::ZeroCopyInputStream *input =
      new google::protobuf::io::ArrayInputStream(buffer, size);
  google::protobuf::TextFormat::Merge(input, _nbla);
  delete input;
  return true;
}

bool _proto_internal::add_protobuf(std::string filename) {
  NNablaProtoBuf param;
  std::ifstream ifs(filename.c_str());
  param.ParseFromIstream(&ifs);
  _nbla->MergeFrom(param);
  return true;
}

bool _proto_internal::add_protobuf(char *buffer, int size) {
  NNablaProtoBuf param;
  std::string buf(buffer);
  param.ParseFromString(buf);
  _nbla->MergeFrom(param);
  return true;
}

bool _proto_internal::add_hdf5(char *buffer, int size) {
  hid_t id = H5LTopen_file_image(buffer, size, H5LT_FILE_IMAGE_DONT_RELEASE);
  if (id >= 0) {
    root = H5Gopen(id, "/", H5P_DEFAULT);
    if (root >= 0) {
      return parse_hdf5_group(root);
    }
  }
  return false;
}

int _proto_internal::num_of_executors() { return _nbla->executor_size(); }

std::vector<std::string> _proto_internal::get_executor_input_names(int index) {
  const Executor &exec = _nbla->executor(index);
  assert(network.name() == "");
  std::vector<std::string> names;
  for (int i = 0; i < exec.data_variable_size(); i++) {
    names.push_back(exec.data_variable(i).variable_name());
  }
  for (int i = 0; i < exec.generator_variable_size(); i++) {
    names.push_back(exec.generator_variable(i).variable_name());
  }
  return names;
}

std::vector<nbla::CgVariablePtr>
_proto_internal::get_executor_input_variables(int index) {
  const Executor &exec = _nbla->executor(index);
  const Network &network = search_network(exec.network_name());
  assert(network.name() == "");
  std::vector<nbla::CgVariablePtr> inputs;
  for (int i = 0; i < exec.data_variable_size(); i++) {
    inputs.push_back(
        create_cgvariable(network, exec.data_variable(i).variable_name()));
  }
  for (int i = 0; i < exec.generator_variable_size(); i++) {
    inputs.push_back(
        create_cgvariable(network, exec.generator_variable(i).variable_name()));
  }
  return inputs;
}

std::vector<nbla::CgVariablePtr>
_proto_internal::get_executor(int index,
                              std::vector<nbla::CgVariablePtr> inputs) {
  const Executor &exec = _nbla->executor(index);

  const Network &original_network = search_network(exec.network_name());
  assert(network.name() == "");
  Network network = expand_network(original_network);

  std::map<std::string, shared_ptr<nbla::CgVariable>> variables;

  int input_index = 0;
  for (int i = 0; i < exec.data_variable_size(); i++, index++) {
    variables[exec.data_variable(i).variable_name()] = inputs[input_index];
    input_index += 1;
  }
  for (int i = 0; i < exec.generator_variable_size(); i++, index++) {
    variables[exec.generator_variable(i).variable_name()] = inputs[input_index];
  }

  for (int i = 0; i < exec.parameter_variable_size(); i++) {
    ParameterVariable p = exec.parameter_variable(i);
    shared_ptr<nbla::CgVariable> var =
        create_cgvariable(network, p.variable_name());

    Parameter param = search_parameter(p.variable_name());
    assert(param.variable_name() == "");

    auto v = var->variable();
    float *data = v->cast_data_and_get_pointer<float>(_ctx);
    assert(param.data_size() == v.get()->size());
    for (int k = 0; k < v.get()->size(); k++) {
      data[k] = param.data(k);
    }
    variables[p.variable_name()] = var;
  }

  std::vector<nbla::CgVariablePtr> n;

  for (int i = 0; i < network.function_size(); i++) {
    Function func = network.function(i);

    NBLA_LOG_INFO("      function name:{} type:{}", func.name(), func.type());

    std::vector<nbla::CgVariablePtr> finputs;
    for (int j = 0; j < func.input_size(); j++) {
      finputs.push_back(variables[func.input(j)]);
    }
    auto cgfunc = create_cgfunction(func);

    if (cgfunc.get() == nullptr) {
      using namespace nbla;
      NBLA_ERROR(error_code::not_implemented,
                 "Function [%s] does not supported yet", func.name().c_str());
    }

    n = nbla::connect(cgfunc, finputs, func.output_size());

    for (int j = 0; j < func.output_size(); j++) {
      variables[func.output(j)] = n[j];
    }
  }

  return n;
}
}
}
