// Copyright 2019,2020,2021 Sony Corporation.
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

// common_fixture.cpp

#include <parametric_functions/common_fixture.hpp>

vector<shared_ptr<Initializer>> create_init_list() {
  vector<shared_ptr<Initializer>> v;

  v.push_back(nullptr);

  auto c1 = make_shared<ConstantInitializer>(1);
  v.push_back(c1);

  return v;
}