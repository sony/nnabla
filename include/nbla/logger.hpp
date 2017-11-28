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

// -*- coding:utf-8 -*-
#ifndef __NBLA_LOGGER_HPP__
#define __NBLA_LOGGER_HPP__

#include <spdlog/spdlog.h>

#include <nbla/defs.hpp>

NBLA_API std::shared_ptr<spdlog::logger> get_logger(void);

#define NBLA_LOG_TRACE(...)                                                    \
  { get_logger()->trace(__VA_ARGS__); }
#define NBLA_LOG_DEBUG(...)                                                    \
  { get_logger()->debug(__VA_ARGS__); }
#define NBLA_LOG_INFO(...)                                                     \
  { get_logger()->info(__VA_ARGS__); }
#define NBLA_LOG_WARN(...)                                                     \
  { get_logger()->warn(__VA_ARGS__); }
#define NBLA_LOG_ERROR(...)                                                    \
  { get_logger()->error(__VA_ARGS__); }
#define NBLA_LOG_CRITICAL(...)                                                 \
  { get_logger()->critical(__VA_ARGS__); }

#ifndef WHOAMI
#include <cstdio>
#define WHOAMI(...)                                                            \
  {                                                                            \
    printf("%s:%d :", __FILE__, __LINE__);                                     \
    printf(__VA_ARGS__);                                                       \
    fflush(stdout);                                                            \
  }
#endif // WHOAMI

#endif
