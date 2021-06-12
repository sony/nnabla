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

#ifndef NBLA_CLI_NBLA_INFER_HPP_
#define NBLA_CLI_NBLA_INFER_HPP_

#include <nbla_utils/nnp.hpp>
NBLA_API bool nbla_infer_core(nbla::Context ctx, int argc, char *argv[]);

#endif // NBLA_CLI_NBLA_INFER_HPP_
