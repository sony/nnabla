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

#ifndef NBLA_UTILS_NNP_NETWORK_EXPANDER_HPP_
#define NBLA_UTILS_NNP_NETWORK_EXPANDER_HPP_

#include "nnabla.pb.h"

#include <list>
#include <unordered_map>

namespace nbla {
namespace utils {
namespace nnp {

enum class VisitState { VISITING = 1, VISITED = 2 };

class NetworkExpander {

public:
  NetworkExpander(const ::Network &network);
  ~NetworkExpander();
  ::Network execute();

private:
  void visit(const ::Function &function);
  void sort_functions();
  ::Network expand_repeat(const ::Network &network);
  string gen_repeat_name(const string &name, const string &id, int index,
                         bool old_naming = false);

private:
  ::Network expand_;
  const ::Network &network_;
  std::list<::Function> sorted_;
  std::unordered_map<std::string, VisitState> visitor_state_;
  bool old_naming_;
  std::map<std::string, vector<std::string>> param_original_names_;
  std::map<std::string, vector<std::string>> delay_var_;
  std::map<std::string, vector<std::string>> repeat_end_var_;
  std::map<std::string, vector<std::string>> repeat_start_var_;
};

template <typename T> bool is_in(std::string k, const T &m) {
  return (m.find(k) != m.end());
};

template <typename T> bool is_not_in(string k, const T &m) {
  return (m.find(k) == m.end());
};

template <typename T> bool search_repeat_id(T m, string id) {
  auto r = m.repeat_id();
  for (auto it = r.begin(); it != r.end(); ++it) {
    if (*it == id) {
      return true;
    }
  }
  return false;
}

template <typename T> void delete_repeat_id(T *v, string id) {
  auto repeat_ids = v->mutable_repeat_id();
  if (repeat_ids == nullptr) {
    return;
  }
  auto it = repeat_ids->begin();
  for (; it != repeat_ids->end(); ++it) {
    if (*it == id) {
      break;
    }
  }
  if (it != repeat_ids->end()) {
    repeat_ids->erase(it);
  }
}

#ifdef DEBUG_NETWORK_EXPANDER
void dump_proto_network(const ::Network &network);
#endif

} // namespace nnp
} // namespace utils
} // namespace nbla

#endif
