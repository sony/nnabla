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
// GlobalConfig
// ----------------------------------------------------------------------
GlobalConfigImpl::GlobalConfigImpl(const ::GlobalConfig &global_config)
    : global_config_proto_(global_config) {
  default_context_ = create_context(global_config.default_context());
}

shared_ptr<nbla::Context>
GlobalConfigImpl::create_context(const ::Context &ctx) {
  vector<string> backends;
  backends.insert(backends.begin(), ctx.backends().begin(),
                  ctx.backends().end());
  return shared_ptr<nbla::Context>(
      new Context(backends, ctx.array_class(), ctx.device_id()));
}

shared_ptr<nbla::Context> GlobalConfigImpl::default_context() {
  return default_context_;
}

// ----------------------------------------------------------------------
// TrainingConfig
// ----------------------------------------------------------------------
TrainingConfigImpl::TrainingConfigImpl(const ::TrainingConfig &training_config)
    : training_config_proto_(training_config) {}

const long long int TrainingConfigImpl::max_epoch() const {
  return training_config_proto_.max_epoch();
}

const long long int TrainingConfigImpl::iter_per_epoch() const {
  return training_config_proto_.iter_per_epoch();
}

const bool TrainingConfigImpl::save_best() const {
  return training_config_proto_.save_best();
}
}
}
}
