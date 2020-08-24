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

#include <nbla/logger.hpp>
#include <nbla_utils/nnp.hpp>

#include <fstream>
#include <iostream>

#include "nbla_commands.hpp"

#include <nbla_utils/nnp.hpp>
#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>
#endif
using namespace nbla;
using namespace utils;
using namespace nnp;
using namespace std;

/******************************************/
// nnp training command sample
/******************************************/
#ifndef _WIN32
bool nbla_train_core(nbla::Context ctx, int argc, char *argv[]) {

  // usage
  if (argc != 3) {
    std::cerr << std::endl;
    std::cerr << "Usage: " << argv[0] << " model.nnp result_dir" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  model.nnp   : model file created by console "
                 "and modified max_epoch and dataset cache_dir"
              << std::endl;
    std::cerr << std::endl;
    return false;
  }

  // get filenames of model and dataset
  const std::string nnp_file(argv[1]);
  const std::string result_dir(argv[2]);

  // create output directory
  struct stat st = {0};
  if (stat(result_dir.c_str(), &st) == -1) {
    mkdir(result_dir.c_str(), 0756);
  }

  // Create a Nnp object
  nbla::utils::nnp::Nnp nnp(ctx);

  // Set nnp file to Nnp object.
  if (!nnp.add(nnp_file)) {
    std::cerr << "Error: not found " << nnp_file << std::endl;
    return false;
  }

  // Training setting
  shared_ptr<TrainingConfig> training_config = nnp.get_training_config();
  const int iter_per_epoch = training_config->iter_per_epoch();
  const int max_epoch = training_config->max_epoch();
  const int max_iter = iter_per_epoch * max_epoch;

  // training loop
  string monitoring_report = result_dir + "/monitoring_report.yaml";
  std::unique_ptr<FILE, int (*)(FILE *)> fmon(
      fopen(monitoring_report.c_str(), "w"), fclose);
  if (fmon.get() == NULL) {
    std::cerr << "Error: Could not open monitoring_report.yaml" << std::endl;
    return false;
  }

  // Create optimizer
  vector<shared_ptr<nbla::utils::nnp::Optimizer>> os;
  for (auto optimizer_name : nnp.get_optimizer_names()) {
    os.push_back(nnp.get_optimizer(optimizer_name));
  }

  // Create monitor
  vector<shared_ptr<nbla::utils::nnp::Monitor>> ms;
  for (auto monitor_name : nnp.get_monitor_names()) {
    ms.push_back(nnp.get_monitor(monitor_name));
  }

  float cost = 0.0;
  int epoch = 0;
  for (int iter = 0; iter < max_iter; iter++) {

    for (auto optimizer : os) {
      cost += optimizer->update(iter);
    }

    if (!((iter + 1) % iter_per_epoch == 0))
      continue;

    cost /= iter_per_epoch;
    fprintf(fmon.get(), "%d:\n", epoch);
    fprintf(fmon.get(), "  cost: %.16f\n", cost);
    fprintf(stdout, "epoch: %3d cost: %f ", epoch, cost);
    cost = 0;
    epoch++;

    for (auto monitor : ms) {
      float monitor_acc = monitor->monitor_epoch();
      fprintf(fmon.get(), "  %s: %.16f\n", monitor->name().c_str(),
              monitor_acc);
      fprintf(stdout, "%s: %f ", monitor->name().c_str(), monitor_acc);
    }
    fprintf(stdout, "\n");
  }

#ifdef NBLA_UTILS_WITH_HDF5
  string parameter_file = result_dir + "/parameters.h5";
#else
  string parameter_file = result_dir + "/parameters.protobuf";
#endif
  if (!nnp.save_parameters(parameter_file.c_str())) {
    return false;
  }

  return true;
}

bool nbla_train(int argc, char *argv[]) {
  // Create a context
  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Train
  if (!nbla_train_core(ctx, argc, argv)) {
    return false;
  }

  return true;
}

#else
bool nbla_train(int argc, char *argv[]) {
  std::cerr << "Error: nbla train is not supported for windows." << std::endl;
  return false;
}
#endif
