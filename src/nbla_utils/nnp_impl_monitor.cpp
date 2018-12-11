// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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
#include <iostream>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/function/mean.hpp>
#include <nbla/function/sink.hpp>

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
// MonitorImpl
// ----------------------------------------------------------------------
const nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

MonitorImpl::MonitorImpl(const nbla::Context &ctx, const ::Monitor &monitor,
                         shared_ptr<Network> network,
                         shared_ptr<DatasetImpl> dataset)
    : ctx_(ctx), monitor_proto_(monitor), network_(network), dataset_(dataset) {
  data_iterator_ = shared_ptr<DataIteratorFromCacheFiles>(
      new DataIteratorFromCacheFiles(dataset));
  network_->set_batch_size(data_iterator_->get_batch_size());
}

string MonitorImpl::name() const { return monitor_proto_.name(); }
string MonitorImpl::network_name() const {
  return monitor_proto_.network_name();
}
string MonitorImpl::dataset_name() const {
  if (monitor_proto_.dataset_name_size() != 1) {
    NBLA_ERROR(error_code::value, "Currently only one dataset supported.");
  }
  return monitor_proto_.dataset_name()[0];
}

vector<Monitor::DataVariable> MonitorImpl::get_data_variables() {
  vector<Monitor::DataVariable> ret;
  for (auto it = monitor_proto_.data_variable().begin();
       it != monitor_proto_.data_variable().end(); it++) {
    Monitor::DataVariable v{it->variable_name(), it->data_name(),
                            network_->get_variable(it->variable_name())};
    ret.push_back(v);
  }
  return ret;
}

vector<Monitor::MonitorVariable> MonitorImpl::get_monitor_variables() {
  vector<Monitor::MonitorVariable> ret;
  for (auto it = monitor_proto_.monitor_variable().begin();
       it != monitor_proto_.monitor_variable().end(); it++) {
    Monitor::MonitorVariable v{it->variable_name(), it->type(), it->data_name(),
                               it->multiplier(),
                               network_->get_variable(it->variable_name())};
    ret.push_back(v);
  }
  NBLA_CHECK(ret.size() > 0, error_code::value,
             "Monitor `%s`'s output is empty.", name().c_str());
  return ret;
}

shared_ptr<Network> MonitorImpl::get_network() { return network_; }

void MonitorImpl::monitor(const nbla::Context &ctx) {
  vector<Monitor::MonitorVariable> monitor_variables = get_monitor_variables();
  for (int k = 0; k < monitor_variables.size(); k++) {
    Monitor::MonitorVariable x = monitor_variables[k];
    std::cout << "- monitor[" << k << "].name: " << x.data_name << std::endl;
    std::cout << "- monitor[" << k << "].var: " << x.variable_name << std::endl;
    VariablePtr v = x.variable->variable();
    float *values = v->cast_data_and_get_pointer<float>(ctx);
    for (int u = 0; u < v->size(); u++) {
      std::cout << "-- value:          " << values[u] << std::endl;
    }
  }
}

const float MonitorImpl::monitor_epoch() {

  float monitor_acc = 0.0;
  const int max_iter = data_iterator_->get_iter_per_epoch();

  for (int iter = 0; iter < max_iter; iter++) {

    auto sink = make_shared<CgFunction>(create_Sink(ctx_, true));

    auto minibatch = data_iterator_->next();
    for (auto x : this->get_data_variables()) {
      VariablePtr v = x.variable->variable();
      v->set_data(minibatch.at(x.data_name));
    }

    vector<CgVariablePtr> m_means;
    for (auto y : this->get_monitor_variables()) {
      const int ndim = y.variable->variable()->ndim();
      vector<int> axes;
      for (int i = 0; i < ndim; i++)
        axes.push_back(i);
      auto mean = make_shared<CgFunction>(create_Mean(ctx_, axes, false));
      m_means.push_back(connect(mean, {y.variable}, 1)[0]);
    }
    CgVariablePtr m_sink = connect(sink, m_means, 1)[0];
    m_sink->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);

    for (auto m_mean : m_means) {
      float_t *values =
          m_mean->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx);
      monitor_acc += values[0];
    }
  }

  monitor_acc /= max_iter;
  return monitor_acc;
}
}
}
}
