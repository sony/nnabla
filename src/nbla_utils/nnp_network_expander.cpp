// Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

#ifdef _WIN32
typedef int ssize_t;
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "nnp_impl.hpp"
#include "nnp_network_expander.hpp"
#include <nbla/computation_graph/computation_graph.hpp>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>
#include <map>
#include <regex>

#include <nbla/function/sink.hpp>
#include <nbla/logger.hpp>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define open _open
#define O_RDONLY _O_RDONLY
#endif

using namespace std;

namespace nbla {
namespace utils {
namespace nnp {

const int MAX_NAME_LEN = 512;

// ----------------------------------------------------------------------
// Helper functions for debugging
// ----------------------------------------------------------------------
#ifdef DEBUG_NETWORK_EXPANDER
void dump_proto_network(const ::Network &network) {
  static int c = 0;
  std::string output;
  google::protobuf::TextFormat::PrintToString(network, &output);
  std::ofstream network_proto_file;
  std::string network_pb_txt = "network_nntxt_" + std::to_string(c++) + ".txt";
  network_proto_file.open(network_pb_txt, std::ios::out);
  network_proto_file << output;
  network_proto_file.close();
}
#endif

// ----------------------------------------------------------------------
// NetworkExpander
// ----------------------------------------------------------------------
NetworkExpander::NetworkExpander(const ::Network &network)
    : expand_(), network_(network), sorted_(), visitor_state_(),
      old_naming_(false), param_original_names_(), delay_var_(),
      repeat_end_var_(), repeat_start_var_() {}

NetworkExpander::~NetworkExpander() {}

void NetworkExpander::visit(const ::Function &func) {
  auto it = visitor_state_.find(func.name());
  if (it == visitor_state_.end()) {
    visitor_state_[func.name()] = VisitState::VISITING;
    for (auto output : func.output()) {
      for (auto next_func : expand_.function()) {
        for (auto input : next_func.input()) {
          if (input == output) {
            visit(next_func);
          }
        }
      }
    }
  } else {
    if (it->second == VisitState::VISITING) {
      cerr << "Fatal error! network is not DAG!" << endl;
      exit(-1);
    }
    // otherwise, visit state is VISITED
    return;
  }
  visitor_state_[func.name()] = VisitState::VISITED;
  sorted_.push_front(func);
}

void NetworkExpander::sort_functions() {
  for (auto func : expand_.function()) {
    if (is_not_in(func.name(), visitor_state_)) {
      visit(func);
    }
  }
}

string NetworkExpander::gen_repeat_name(const string &name, const string &id,
                                        int index, bool old_naming) {
  if (old_naming) {
    regex e("\\{" + id + "\\}");
    return regex_replace(name, e, "_" + to_string(index));
  } else {
    char buffer[MAX_NAME_LEN];
    ::snprintf(buffer, sizeof(buffer), "%s_%s_%d", name.c_str(), id.c_str(),
               index);
    return string(buffer);
  }
}

::Network NetworkExpander::expand_repeat(const ::Network &network) {
  if (network.repeat_info_size() == 0)
    return network;

  ::Network net;
  net.CopyFrom(network);
  auto ri = net.repeat_info(0);
  auto repeat_info = net.mutable_repeat_info();
  repeat_info->erase(repeat_info->begin());
  net.clear_variable();
  for (auto v : network.variable()) {
    if (v.type() == "Parameter") {
      if (is_not_in(v.name(), param_original_names_)) {
        param_original_names_[v.name()] = vector<string>();
      }
    }
    if (search_repeat_id(v, ri.id())) {
      for (int i = 0; i < ri.times(); ++i) {
        string name;
        if (v.type() == "Parameter") {
          name = gen_repeat_name(v.name(), ri.id(), i, true);
          param_original_names_[v.name()].push_back(name);
        } else {
          name = gen_repeat_name(v.name(), ri.id(), i);
        }
        auto var = net.add_variable();
        var->CopyFrom(v);
        var->set_name(name);
        delete_repeat_id(var, ri.id());
      }
    } else {
      if ((v.type() == "Parameter") && (v.repeat_id_size() == 0) &&
          (param_original_names_[v.name()].size() == 0))
        param_original_names_[v.name()].push_back(v.name());
      auto var = net.add_variable();
      var->CopyFrom(v);
    }
  }

  delay_var_.clear();
  for (auto f : network.function()) {
    if (f.type() == "Delay") {
      if (f.recurrent_param().repeat_id() == ri.id()) {
        auto output = f.output(0);
        delay_var_[output] = vector<string>(0);
        for (int i = 0; i < ri.times(); ++i) {
          if (i == 0) {
            delay_var_[output].push_back(f.input(1));
          } else {
            auto name = gen_repeat_name(f.input(0), ri.id(), i - 1);
            delay_var_[output].push_back(name);
          }
        }
      }
    }
  }

  repeat_end_var_.clear();
  for (auto f : network.function()) {
    if (f.type() == "RepeatEnd") {
      if (f.repeat_param().repeat_id() == ri.id()) {
        auto output = f.output(0);
        repeat_end_var_[output] = vector<string>(0);
        for (int i = 0; i < ri.times(); ++i) {
          auto name = gen_repeat_name(f.input(0), ri.id(), i);
          repeat_end_var_[output].push_back(name);
        }
      }
    }
  }

  repeat_start_var_.clear();
  for (auto f : network.function()) {
    if (f.type() == "RepeatStart") {
      if (f.repeat_param().repeat_id() == ri.id()) {
        auto output = f.output(0);
        repeat_start_var_[output] = vector<string>(0);
        for (int i = 0; i < ri.times(); ++i) {
          if (i == 0) {
            auto v = f.input(0);
            if (is_in(v, repeat_end_var_)) {
              v = repeat_end_var_[v][ri.times() - 1];
            }
            repeat_start_var_[output].push_back(v);
          } else {
            auto v = f.input(1);
            if (is_in(v, repeat_end_var_)) {
              v = repeat_end_var_[v][i - 1];
            } else {
              v = gen_repeat_name(v, ri.id(), i - 1);
            }
            repeat_start_var_[output].push_back(v);
          }
        }
      }
    }
  }

  net.clear_function();
  for (auto f : network.function()) {
    if ((f.type() == "RepeatStart") || (f.type() == "RepeatEnd")) {
      if (f.repeat_param().repeat_id() == ri.id()) {
        continue;
      }
    }

    if (f.type() == "Delay") {
      if (f.recurrent_param().repeat_id() == ri.id()) {
        continue;
      }
    }

    if (f.type() == "RecurrentInput") {
      if (f.recurrent_param().repeat_id() == ri.id()) {
        ::Function *func = net.add_function();
        func->CopyFrom(f);
        func->set_type("Split");
        func->mutable_split_param()->set_axis(f.recurrent_param().axis());

        func->clear_output();
        for (int i = 0; i < ri.times(); ++i) {
          auto name = gen_repeat_name(f.output(0), ri.id(), i);
          func->add_output(name);
        }

        delete_repeat_id(func, ri.id());
        func->clear_recurrent_param();
        continue;
      }
    }

    if (f.type() == "RecurrentOutput") {
      if (f.recurrent_param().repeat_id() == ri.id()) {
        ::Function *func = net.add_function();
        func->CopyFrom(f);
        func->set_type("Stack");
        func->mutable_stack_param()->set_axis(f.recurrent_param().axis());
        func->clear_input();
        for (int i = 0; i < ri.times(); ++i) {
          auto name = gen_repeat_name(f.input(0), ri.id(), i);
          func->add_input(name);
        }
        func->clear_recurrent_param();
        continue;
      }
    }

    if (search_repeat_id(f, ri.id())) {
      for (int i = 0; i < ri.times(); ++i) {
        ::Function *func = net.add_function();
        func->CopyFrom(f);
        delete_repeat_id(func, ri.id());

        func->set_name(gen_repeat_name(f.name(), ri.id(), i));

        for (int j = 0; j < func->input_size(); ++j) {
          string vname;
          auto input_name = func->input(j);
          if (is_in(input_name, param_original_names_)) {
            auto name_list = param_original_names_[input_name];
            if (name_list.size() == ri.times()) {
              vname = name_list[i];
            } else {
              vname = input_name;
            }
          } else if (is_in(input_name, repeat_start_var_)) {
            vname = repeat_start_var_[input_name][i];
          } else if (is_in(input_name, repeat_end_var_)) {
            vname = repeat_end_var_[input_name][i];
          } else if (is_in(input_name, delay_var_)) {
            vname = delay_var_[input_name][i];
          } else {
            vname = gen_repeat_name(input_name, ri.id(), i);
          }
          func->set_input(j, vname);
        }

        for (int j = 0; j < f.output_size(); ++j) {
          string vname = gen_repeat_name(f.output(j), ri.id(), i);
          func->set_output(j, vname);
        }
      }
    } else {
      ::Function *func = net.add_function();
      func->CopyFrom(f);
      for (int i = 0; i < f.input_size(); ++i) {
        string input_name = f.input(i);
        if (is_in(input_name, repeat_end_var_)) {
          string vname = repeat_end_var_[input_name][ri.times() - 1];
          func->set_input(i, vname);
        }
      }
    }
  }

  return expand_repeat(net);
}

::Network NetworkExpander::execute() {
  ::Network net;
  expand_ = expand_repeat(network_);
  sort_functions();
  net.CopyFrom(expand_);
  net.clear_function();
  for (auto f : sorted_) {
    ::Function *func = net.add_function();
    func->CopyFrom(f);
  }

  return net;
}

} // namespace nnp
} // namespace utils
} // namespace nbla