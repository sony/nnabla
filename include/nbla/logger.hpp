// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

#include <nbla/defs.hpp>

#include <cstdio>

#define _NBLA_LOG_STDOUT(...)                                                  \
  {                                                                            \
    printf("%s:%d :", __FILE__, __LINE__);                                     \
    printf(__VA_ARGS__);                                                       \
    fflush(stdout);                                                            \
  }

#define _NBLA_LOG_NONE(...)                                                    \
  {}

#define NBLA_LOG_TRACE _NBLA_LOG_NONE
#define NBLA_LOG_DEBUG _NBLA_LOG_NONE
#define NBLA_LOG_INFO _NBLA_LOG_NONE
#define NBLA_LOG_WARN _NBLA_LOG_NONE
#define NBLA_LOG_ERROR _NBLA_LOG_STDOUT
#define NBLA_LOG_CRITICAL _NBLA_LOG_STDOUT

#endif
