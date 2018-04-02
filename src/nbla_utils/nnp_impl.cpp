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
#include <iostream>
#include <map>
#include <nbla/function/sink.hpp>
#include <nbla/logger.hpp>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define open _open
#define O_RDONLY _O_RDONLY
#endif

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

    std::vector<nbla::CgVariablePtr> f_tmp_outputs;
    for (int j = 0; j < func.output_size(); j++) {
      auto it = variables_.find(func.output(j));
      if (it != variables_.end()) {
        CgVariablePtr cg_v = get_cgvariable_or_create(func.output(j));
        cg_v->set_parent(cgfunc);
        f_tmp_outputs.push_back(cg_v);
      }
    }
    // TODO: It may dangerous if output  variable exists.
    if (f_tmp_outputs.size() == 0) {
      auto foutputs = nbla::connect(cgfunc, finputs, func.output_size());
      for (int j = 0; j < func.output_size(); j++) {
        variables_.insert({func.output(j), foutputs[j]});
      }
    } else {
      nbla::connect(cgfunc, finputs, f_tmp_outputs);
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
  herr_t err = H5Sget_simple_extent_dims(sp, dims, nullptr);
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

int NnpImpl::get_network_repeat_nest_depth(const ::Network &orig) {
  // get max nest depth.
  int max_nest_depth = -1;
  for (int i = 0; i < orig.function_size(); i++) {
    int depth = orig.function(i).repeat_id_size();
    if (depth > max_nest_depth) {
      max_nest_depth = depth;
    }
  }
  return max_nest_depth;
}

std::vector<std::string> NnpImpl::create_suffixes(std::string prefix,
                                                  std::vector<std::string> ids,
                                                  std::vector<int> times) {
  std::vector<std::string> suffixes;
  std::string id = ids.at(0);
  ids.erase(ids.begin());
  int max = times.at(0);
  times.erase(times.begin());
  for (int i = 0; i < max; i++) {
    std::string suffix = prefix + "_" + id + "[" + std::to_string(i) + "]";
    if (ids.size() > 0) {
      auto sub = create_suffixes(suffix, ids, times);
      std::copy(sub.begin(), sub.end(), std::back_inserter(suffixes));
    } else {
      suffixes.push_back(suffix);
    }
  }
  return suffixes;
}

std::vector<std::string>
NnpImpl::create_var_suffixes(std::map<std::string, int> repeat_info,
                             ::Variable var) {
  std::vector<std::string> ids;
  std::vector<int> times;
  for (int j = 0; j < var.repeat_id_size(); j++) {
    auto rid = var.repeat_id(j);
    ids.push_back(rid);
    times.push_back(repeat_info[rid]);
  }
  return create_suffixes("", ids, times);
}

std::vector<std::string>
NnpImpl::create_func_suffixes(std::map<std::string, int> repeat_info,
                              ::Function func) {
  std::vector<std::string> ids;
  std::vector<int> times;
  for (int j = 0; j < func.repeat_id_size(); j++) {
    auto rid = func.repeat_id(j);
    ids.push_back(rid);
    times.push_back(repeat_info[rid]);
  }
  return create_suffixes("", ids, times);
}

::Network NnpImpl::expand_network(const ::Network &orig) {
  ::Network net;
  net.set_name(orig.name());
  net.set_batch_size(orig.batch_size());

  if (get_network_repeat_nest_depth(orig) > 0) {

    std::map<std::string, int> repeat_info;
    for (int i = 0; i < orig.repeat_info_size(); i++) {
      repeat_info[orig.repeat_info(i).id()] = orig.repeat_info(i).times();
    }

    // Prepare variables.
    std::map<std::string, int> vars;
    std::map<std::string, int> ovars;

    int vid = 0;
    for (int i = 0; i < orig.variable_size(); i++) {
      auto ovar = orig.variable(i);
      ovars[ovar.name()] = i;

      if (ovar.repeat_id_size() > 0) {
        auto suffixes = create_var_suffixes(repeat_info, ovar);
        for (int j = 0; j < suffixes.size(); j++) {
          vid = net.variable_size();
          ::Variable *var = net.add_variable();
          var->CopyFrom(ovar);
          var->clear_repeat_id();
          var->set_name(var->name() + suffixes[j]);
          vars[var->name()] = vid;
        }
      } else {
        vid = net.variable_size();
        ::Variable *var = net.add_variable();
        var->CopyFrom(ovar);
        vars[var->name()] = vid;
      }
    } // i

    // Prepare fuctions
    std::map<std::string, int> funcs;
    std::map<std::string, int> ofuncs;

    int fid = 0;
    for (int i = 0; i < orig.function_size(); i++) {
      auto ofunc = orig.function(i);
      auto type = ofunc.type();
      ofuncs[ofunc.name()] = i;

      if (type == "RecurrentInput") {

        // RecurrentInput will convert into 'Split' function.
        ::Function *func = net.add_function();
        fid = net.function_size();
        func->set_name(ofunc.name());
        func->set_type("Split");
        funcs[func->name()] = fid;

        ::SplitParameter *split_param = new ::SplitParameter();
        auto axis = ofunc.recurrent_param().axis();
        split_param->set_axis(axis);
        func->set_allocated_split_param(split_param);

        // Prepare input(s)
        // Just copy from ofunc.
        for (int j = 0; j < ofunc.input_size(); j++) {
          func->add_input(ofunc.input(j));
        }

        // Prepare output(s)
        auto ovar = orig.variable(ovars[ofunc.output(0)]);
        auto suffixes = create_var_suffixes(repeat_info, ovar);
        for (int j = 0; j < suffixes.size(); j++) {
          func->add_output(ofunc.output(0) + suffixes[j]);
        }

      } else if (ofunc.repeat_id_size() > 0) {
        auto suffixes = create_func_suffixes(repeat_info, ofunc);

        for (int j = 0; j < suffixes.size(); j++) {
          auto suffix = suffixes[j];
          fid = net.function_size();
          ::Function *func = net.add_function();
          func->CopyFrom(ofunc);
          func->set_name(func->name() + suffix);
          funcs[func->name()] = fid;

          if (type == "RepeatStart") {
            // Change function type to identity
            func->set_type("Identity");
            // Change input....
            NBLA_CHECK(func->input_size() == 2, error_code::value,
                       "Input size of RepeatStart must be 2. %s (%d != 2)",
                       func->name().c_str(), (int)func->input_size());
            std::string input;
            if (j == 0) {
              input = func->input(0);
            } else {
              input = func->input(1) + suffixes[j - 1];
            }
            func->clear_input();
            func->add_input(input);
          } else if (type == "Delay") {

            // Change function type to identity
            func->set_type("Identity");
            // Change input....
            NBLA_CHECK(func->input_size() == 2, error_code::value,
                       "Input size of Delay must be 2. %s (%d != 2)",
                       func->name().c_str(), (int)func->input_size());
            std::string input;
            if (j == 0) {
              input = func->input(1);
            } else {
              input = func->input(0) + suffixes[j - 1];
            }
            func->clear_input();
            func->add_input(input);
          } else {
            // Add suffix to input params.
            for (int k = 0; k < func->input_size(); k++) {
              auto inp = func->input(k);
              if (vars.count(inp + suffix) > 0) {
                func->set_input(k, inp + suffix);
              } else {
                func->set_input(k, inp);
              }
            }
          }
          // Add suffix to output params.
          for (int k = 0; k < func->output_size(); k++) {
            auto out = func->output(k);
            if (vars.count(out + suffix) > 0) {
              func->set_output(k, out + suffix);
            } else {
              func->set_output(k, out);
            }
          }
        }

      } else {
        fid = net.function_size();
        ::Function *func = net.add_function();
        func->CopyFrom(ofunc);
        funcs[func->name()] = fid;

        if (type == "RepeatEnd") {
          // Change function type to identity
          func->set_type("Identity");
          // Add suffix to input params.
          for (int j = 0; j < func->input_size(); j++) {
            auto inp = func->input(j);
            auto ovar = ovars[func->input(j)];
            auto suffixes =
                create_var_suffixes(repeat_info, orig.variable(ovar));
            func->set_input(j, inp + suffixes[suffixes.size() - 1]);
          }
        } else if (type == "RecurrentOutput") {
          // RecurrentOutput will convert into 'Stack' function.
          func->set_name(ofunc.name());
          func->set_type("Stack");
          ::StackParameter *stack_param = new ::StackParameter();
          auto axis = ofunc.recurrent_param().axis();
          stack_param->set_axis(axis);
          func->set_allocated_stack_param(stack_param);

          // Prepare input(s)
          auto ovar = orig.variable(ovars[ofunc.input(0)]);
          auto suffixes = create_var_suffixes(repeat_info, ovar);

          auto input_size = func->input_size();
          for (int j = 0; j < suffixes.size(); j++) {
            if (j < input_size) {
              func->set_input(j, ofunc.input(0) + suffixes[j]);
            } else {
              func->add_input(ofunc.input(0) + suffixes[j]);
            }
          }
        }
      }

    } // i

  } else {
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
    NBLA_LOG_INFO("Initial data of {} was found.", it->name());
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
