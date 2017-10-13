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

#include "nnp_impl.hpp"
#include <nbla/computation_graph/computation_graph.hpp>

#include <fstream>
#include <map>
#include <nbla/function/sink.hpp>
#include <nbla/logger.hpp>

namespace nbla {
namespace utils {
namespace nnp {

// ----------------------------------------------------------------------
// NetworkImpl
// ----------------------------------------------------------------------

NetworkImpl::NetworkImpl(const nbla::Context &ctx, const ::Network &network,
                         const unordered_map<string, CgVariablePtr> &parameters)
    : ctx_(ctx), network_proto_(network), parameters_(parameters) {

  // Create hash map for faster query of proto Variable.
  for (auto it = network_proto_.variable().begin();
       it != network_proto_.variable().end(); it++) {
    variable_protos_.insert({it->name(), &(*it)});
  }
}

void NetworkImpl::build() {
  variables_.clear();
  variables_.insert(parameters_.begin(), parameters_.end());
  for (int i = 0; i < network_proto_.function_size(); i++) {
    ::Function func = network_proto_.function(i);

    NBLA_LOG_INFO("      function name:{} type:{}", func.name(), func.type());

    std::vector<nbla::CgVariablePtr> finputs;
    for (auto inp = func.input().begin(); inp != func.input().end(); inp++) {
      CgVariablePtr cg_v = get_cgvariable_or_create(*inp);
      finputs.push_back(cg_v);
    }
    auto cgfunc = create_cgfunction(func);

    if (cgfunc.get() == nullptr) {
      using namespace nbla;
      NBLA_ERROR(error_code::not_implemented,
                 "Function [%s] is not supported yet", func.name().c_str());
    }

    auto foutputs = nbla::connect(cgfunc, finputs, func.output_size());

    for (int j = 0; j < func.output_size(); j++) {
      variables_.insert({func.output(j), foutputs[j]});
    }
  }
}

shared_ptr<nbla::CgVariable>
NetworkImpl::get_cgvariable_or_create(const string &name) {
  // Use variable set by replace_variable
  auto it_r = replace_var_list_.find(name);
  if (it_r != replace_var_list_.end()) {
    return it_r->second;
  }
  // Use variable set by replace_variable
  auto it = variables_.find(name);
  if (it != variables_.end()) {
    return it->second;
  }
  auto var_it = variable_protos_.find(name);
  NBLA_CHECK(
      var_it != variable_protos_.end(), error_code::value,
      "%s could not be found in variable_protos_. This does not usualy happen.",
      name.c_str());
  const ::Variable *var = var_it->second;
  // Create a new on and register to variables_.
  // Create shape
  nbla::Shape_t shape(var->shape().dim().begin(), var->shape().dim().end());
  if (shape[0] == -1) {
    shape[0] = batch_size();
  }
  // TODO: set need_grad
  auto cg_v = std::make_shared<nbla::CgVariable>(shape, false);
  // Register variable
  variables_.insert({name, cg_v});
  return cg_v;
}

void NetworkImpl::replace_variable(const string &name, CgVariablePtr variable) {
  require_build_ = true;
  replace_var_list_.insert({name, variable});
}

CgVariablePtr NetworkImpl::get_variable(const string &name) {
  if (require_build_) {
    build();
    require_build_ = false;
  }
  auto it = variables_.find(name);
  assert(it != variables_.end());
  return it->second;
}

string NetworkImpl::name() const { return network_proto_.name(); }

void NetworkImpl::set_batch_size(int batch_size) {
  if (batch_size_ <= 0 || batch_size_ != batch_size) {
    require_build_ = true;
  }
  batch_size_ = batch_size;
}

int NetworkImpl::batch_size() const {
  if (batch_size_ > 0) {
    return batch_size_;
  }
  assert(network_proto_.batch_size() > 0);
  return network_proto_.batch_size();
}
// ----------------------------------------------------------------------
// ExecutorImpl
// ----------------------------------------------------------------------
ExecutorImpl::ExecutorImpl(const ::Executor &executor,
                           shared_ptr<Network> network)
    : executor_proto_(executor), network_(network) {}

void ExecutorImpl::update_sink() {
  auto outputs = get_output_variables();
  if (outputs.size() == 1) {
    sink_ = outputs[0].variable;
    return;
  }
  vector<CgVariablePtr> inputs;
  for (auto it = outputs.begin(); it != outputs.end(); it++) {
    inputs.push_back(it->variable);
  }
  auto f = make_shared<CgFunction>(create_Sink(nbla::Context(), true));
  sink_ = nbla::connect(f, inputs, 1)[0];
}

string ExecutorImpl::name() const { return executor_proto_.name(); }

string ExecutorImpl::network_name() const {
  return executor_proto_.network_name();
}

void ExecutorImpl::set_batch_size(int batch_size) {
  if (batch_size == network_->batch_size()) {
    return;
  }
  network_->set_batch_size(batch_size);
  sink_ = nullptr;
}

int ExecutorImpl::batch_size() const { return network_->batch_size(); }

vector<Executor::DataVariable> ExecutorImpl::get_data_variables() {
  vector<Executor::DataVariable> ret;
  for (auto it = executor_proto_.data_variable().begin();
       it != executor_proto_.data_variable().end(); it++) {
    Executor::DataVariable v{it->variable_name(), it->data_name(),
                             network_->get_variable(it->variable_name())};
    ret.push_back(v);
  }
  return ret;
}

vector<Executor::OutputVariable> ExecutorImpl::get_output_variables() {
  vector<Executor::OutputVariable> ret;
  for (auto it = executor_proto_.output_variable().begin();
       it != executor_proto_.output_variable().end(); it++) {
    Executor::OutputVariable v{it->variable_name(), it->type(), it->data_name(),
                               network_->get_variable(it->variable_name())};
    ret.push_back(v);
  }
  NBLA_CHECK(ret.size() > 0, error_code::value,
             "Executor `%s`'s output is empty.", name().c_str());
  return ret;
}

shared_ptr<Network> ExecutorImpl::get_network() { return network_; }

void ExecutorImpl::execute() {
  if (sink_ == nullptr) {
    update_sink();
  }
  sink_->forward(true, false);
}

// ----------------------------------------------------------------------
// NnpImpl
// ----------------------------------------------------------------------
NnpImpl::NnpImpl(const nbla::Context &ctx)
    : ctx_(ctx), proto_(new NNablaProtoBuf()) {}

#ifdef NBLA_UTILS_WITH_HDF5
bool NnpImpl::parse_hdf5_dataset(std::string name, hid_t did) {
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
  // TODO: Other data types than float.
  err = H5Dread(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  if (err >= 0) {
    Shape_t shape(dims, dims + rank);
    // TODO: Set need_grad.
    CgVariablePtr cg_v = std::make_shared<CgVariable>(shape, false);
    float *data =
        cg_v->variable()->template cast_data_and_get_pointer<float>(kCpuCtx);
    for (int i = 0; i < size / sizeof(float); i++) {
      data[i] = buffer[i];
    }
    parameters_.insert({variable_name, cg_v});
    return true;
  }
  return false;
  NBLA_ERROR(error_code::not_implemented, "HDF5 is not enabled when build.");
  return false;
}

bool NnpImpl::parse_hdf5_group(hid_t gid) {
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
#endif

::Network NnpImpl::expand_network(const ::Network &orig) {
  ::Network net;
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
      ::RepeatInfo *rep = net.add_repeat_info();
      rep->CopyFrom(orig.repeat_info(i));
      // WHOAMI("%d %s\n", i, rep->id().c_str());
    }

    for (int i = 0; i < orig.variable_size(); i++) {
      ::Variable *var = net.add_variable();
      var->CopyFrom(orig.variable(i));
      // WHOAMI("%s\n", (var->name()).c_str());
    }

    for (int i = 0; i < orig.function_size(); i++) {
      const ::Function &orig_func = orig.function(i);
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
        ::Function *func = net.add_function();
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

const ::Network &NnpImpl::search_network(std::string name) {
  NBLA_LOG_INFO("    Searching net {}", name);
  for (int i = 0; i < proto_->network_size(); i++) {
    if (proto_->network(i).name() == name) {
      NBLA_LOG_INFO("      Found at {}", i);
      return proto_->network(i);
    }
  }
  static const ::Network null_net;
  return null_net;
}

void NnpImpl::update_parameters() {
  for (auto it = proto_->parameter().begin(); it != proto_->parameter().end();
       it++) {
    const string &name = it->variable_name();
    Shape_t shape(it->shape().dim().begin(), it->shape().dim().end());
    // TODO: set need_grad
    CgVariablePtr cg_v = std::make_shared<CgVariable>(shape, false);
    float *data =
        cg_v->variable()->template cast_data_and_get_pointer<float>(kCpuCtx);
    auto &p_data = it->data();
    NBLA_CHECK(p_data.size() == cg_v->variable()->size(), error_code::value,
               "Inconsistent size in proto parameter %s (%d != %d)",
               name.c_str(), (int)p_data.size(), (int)cg_v->variable()->size());
    for (int i = 0; i < p_data.size(); i++) {
      data[i] = p_data[i];
    }
    parameters_.insert({name, cg_v});
  }
  proto_->clear_parameter(); // Reset all parameters consumed.
}

bool NnpImpl::add_prototxt(std::string filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  google::protobuf::io::ZeroCopyInputStream *input =
      new google::protobuf::io::FileInputStream(fd);
  google::protobuf::TextFormat::Merge(input, proto_.get());
  delete input;
  close(fd);
  update_parameters();
  return true;
}

bool NnpImpl::add_prototxt(char *buffer, int size) {
  google::protobuf::io::ZeroCopyInputStream *input =
      new google::protobuf::io::ArrayInputStream(buffer, size);
  google::protobuf::TextFormat::Merge(input, proto_.get());
  delete input;
  update_parameters();
  return true;
}

bool NnpImpl::add_protobuf(std::string filename) {
  NNablaProtoBuf param;
  std::ifstream ifs(filename.c_str());
  param.ParseFromIstream(&ifs);
  proto_->MergeFrom(param);
  update_parameters();
  return true;
}

bool NnpImpl::add_protobuf(char *buffer, int size) {
  NNablaProtoBuf param;
  std::string buf(buffer, size);
  param.ParseFromString(buf);
  proto_->MergeFrom(param);
  update_parameters();
  return true;
}

bool NnpImpl::add_hdf5(char *buffer, int size) {
#ifdef NBLA_UTILS_WITH_HDF5
  hid_t id = H5LTopen_file_image(buffer, size, H5LT_FILE_IMAGE_DONT_RELEASE);
  if (id >= 0) {
    root_ = H5Gopen(id, "/", H5P_DEFAULT);
    if (root_ >= 0) {
      return parse_hdf5_group(root_);
    }
  }
#else
  NBLA_LOG_WARN("HDF5 not enabled during build.");
#endif
  return false;
}

vector<string> NnpImpl::get_network_names() {
  vector<string> list;
  for (int i = 0; i < proto_->network_size(); i++) {
    list.push_back(proto_->network(i).name());
  }
  return list;
}

shared_ptr<Network> NnpImpl::get_network(const string &name) {
  // Find network proto
  const ::Network &orig_network = search_network(name);
  NBLA_CHECK(orig_network.name() != "", error_code::value,
             "Network '%s' not found in proto", name.c_str());
  ::Network network = expand_network(orig_network);
  // Filter parameters
  unordered_map<string, CgVariablePtr> parameters;
  for (auto it = network.variable().begin(); it != network.variable().end();
       it++) {
    auto found = parameters_.find(it->name());
    if (found == parameters_.end()) {
      continue;
    }
    parameters.insert({found->first, found->second});
  }
  return shared_ptr<Network>(
      new Network(new NetworkImpl(ctx_, network, parameters)));
}

vector<string> NnpImpl::get_executor_names() {
  vector<string> list;
  for (auto it = proto_->executor().begin(); it != proto_->executor().end();
       it++) {
    list.push_back(it->name());
  }
  return list;
}

shared_ptr<Executor> NnpImpl::get_executor(const string &name) {
  for (auto it = proto_->executor().begin(); it != proto_->executor().end();
       it++) {
    if (it->name() != name) {
      continue;
    }
    return shared_ptr<Executor>(
        new Executor(new ExecutorImpl(*it, get_network(it->network_name()))));
  }
  NBLA_ERROR(error_code::value, "Executor `%s` not found", name.c_str());
}
}
}
}
