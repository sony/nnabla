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
#include <nbla/solver.hpp>
#include <nbla/solver/adabelief.hpp>
#include <nbla/solver/adadelta.hpp>
#include <nbla/solver/adagrad.hpp>
#include <nbla/solver/adam.hpp>
#include <nbla/solver/adamax.hpp>
#include <nbla/solver/momentum.hpp>
#include <nbla/solver/nesterov.hpp>
#include <nbla/solver/rmsprop.hpp>
#include <nbla/solver/rmsprop_graves.hpp>
#include <nbla/solver/sgd.hpp>

namespace nbla {
namespace utils {
namespace nnp {
shared_ptr<nbla::Solver> OptimizerImpl::create_solver(const ::Solver &solver) {
  // Solver creator
  if (solver.type() == "Adadelta") {
    AdadeltaParameter param = solver.adadelta_param();
    return create_AdadeltaSolver(ctx_, param.lr(), param.decay(), param.eps());
  }
  if (solver.type() == "Adagrad") {
    AdagradParameter param = solver.adagrad_param();
    return create_AdagradSolver(ctx_, param.lr(), param.eps());
  }
  if (solver.type() == "AdaBelief") {
    AdaBeliefParameter param = solver.adabelief_param();
    return create_AdaBeliefSolver(ctx_, param.alpha(), param.beta1(),
                                  param.beta2(), param.eps(), param.wd(),
                                  param.amsgrad(), param.weight_decouple(),
                                  param.fixed_decay(), param.rectify());
  }
  if (solver.type() == "Adam") {
    AdamParameter param = solver.adam_param();
    return create_AdamSolver(ctx_, param.alpha(), param.beta1(), param.beta2(),
                             param.eps());
  }
  if (solver.type() == "Adamax") {
    AdamaxParameter param = solver.adamax_param();
    return create_AdamaxSolver(ctx_, param.alpha(), param.beta1(),
                               param.beta2(), param.eps());
  }
  if (solver.type() == "Momentum") {
    MomentumParameter param = solver.momentum_param();
    return create_MomentumSolver(ctx_, param.lr(), param.momentum());
  }
  if (solver.type() == "Nesterov") {
    NesterovParameter param = solver.nesterov_param();
    return create_NesterovSolver(ctx_, param.lr(), param.momentum());
  }
  if (solver.type() == "RMSprop") {
    RMSpropParameter param = solver.rmsprop_param();
    return create_RMSpropSolver(ctx_, param.lr(), param.decay(), param.eps());
  }
  if (solver.type() == "RMSpropGraves") {
    RMSpropGravesParameter param = solver.rmsprop_graves_param();
    return create_RMSpropGravesSolver(ctx_, param.lr(), param.decay(),
                                      param.momentum(), param.eps());
  }
  if (solver.type() == "Sgd") {
    SgdParameter param = solver.sgd_param();
    return create_SgdSolver(ctx_, param.lr());
  }
  return nullptr;
}
}
}
}