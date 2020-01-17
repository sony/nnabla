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
// test_variable.cpp

#include "gtest/gtest.h"
#include <nbla/common.hpp>
#include <nbla/logger.hpp>

namespace nbla {

TEST(LoggerTest, Logging) {
  int num = 0;
  NBLA_LOG_TRACE("TRACE LOG {}", num++);
  NBLA_LOG_DEBUG("DEBUG LOG {}", num++);
  NBLA_LOG_INFO("INFO LOG {}", num++);
  NBLA_LOG_WARN("WARN LOG {}", num++);
  NBLA_LOG_ERROR("ERROR LOG {}", num++);
  NBLA_LOG_CRITICAL("CRITICAL LOG {}", num++);
}
}
