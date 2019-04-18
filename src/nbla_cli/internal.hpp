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

#ifndef H_INTERNAL_HPP_
#define H_INTERNAL_HPP_

#include <nbla_utils/nnp.hpp>
std::vector<std::string> add_files_to_nnp(nbla::utils::nnp::Nnp &nnp,
                                          std::vector<std::string> files,
                                          bool on_memory = false);

#endif // H_INTERNAL_HPP_
