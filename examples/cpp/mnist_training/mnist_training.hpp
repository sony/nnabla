// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fstream>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>
#include <zlib.h>

using namespace std;

// Include nnabla header files
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/solver/adam.hpp>
#include <nbla_utils/nnp.hpp>

using namespace nbla;
using namespace utils;
using namespace nnp;

/******************************************/
// Example of data provider
/******************************************/

vector<vector<uint8_t>> read_images(const string &data_root,
                                    const string &filename) {

  gzFile fp = gzopen((data_root + filename).c_str(), "rb");
  if (fp == NULL) {
    cerr << "This sample requires mnist data downloaded before." << endl;
    exit(0);
  }

  char header[16];
  gzread(fp, header, 16);
  int num_images = 60000;
  int num_rows = 28;
  int num_cols = 28;

  vector<vector<uint8_t>> images;
  for (int i = 0; i < num_images; ++i) {
    vector<uint8_t> x;
    for (int r = 0; r < num_rows; ++r) {
      for (int c = 0; c < num_cols; ++c) {
        unsigned char temp = 0;
        gzread(fp, (char *)&temp, sizeof(temp));
        x.push_back(temp);
      }
    }
    images.push_back(x);
  }
  return images;
}

vector<uint8_t> read_labels(const string &data_root, const string &filename) {

  gzFile fp = gzopen((data_root + filename).c_str(), "rb");
  if (fp == NULL) {
    cerr << "This sample requires mnist data downloaded before." << endl;
    exit(0);
  }

  char header[8];
  gzread(fp, header, 8);
  int num_images = 60000;

  vector<uint8_t> labels;
  for (int i = 0; i < num_images; ++i) {
    unsigned char temp = 0;
    gzread(fp, (char *)&temp, sizeof(temp));
    labels.push_back(temp);
  }
  return labels;
}

struct Pair {
  std::vector<uint8_t> data;
  int label;
};

class MnistDataIterator {
private:
public:
  MnistDataIterator();
  ~MnistDataIterator();

  vector<Pair> dataset_;
  int current_idx_;
  void permutation();
  const vector<Pair> get_batch(int batchsize);
};

MnistDataIterator::MnistDataIterator() : current_idx_(0) {
  const string image_gz = "train-images-idx3-ubyte.gz";
  const string label_gz = "train-labels-idx1-ubyte.gz";
  vector<vector<uint8_t>> images = read_images("./", image_gz);
  vector<uint8_t> labels = read_labels("./", label_gz);
  for (int i = 0; i < images.size(); i++) {
    Pair xt;
    xt.data = images[i];
    xt.label = labels[i];
    dataset_.push_back(xt);
  }
  permutation();
}

MnistDataIterator::~MnistDataIterator() {}

void MnistDataIterator::permutation() {
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(std::begin(dataset_), std::end(dataset_), g);
}

const vector<Pair> MnistDataIterator::get_batch(int batch_size) {

  const int n_data = dataset_.size();

  vector<Pair> batch_data(batch_size);

  int n_count = 0;
  if (n_data <= current_idx_ + batch_size)
    n_count = n_data - current_idx_;
  for (int i = 0; i < n_count; i++)
    batch_data[i] = dataset_[current_idx_++];

  if (n_data <= current_idx_) {
    permutation();
    current_idx_ = 0;
  }

  for (int i = n_count; i < batch_size; i++)
    batch_data[i] = dataset_[current_idx_++];

  return batch_data;
}

/******************************************/
// Example of mnist training
/******************************************/
bool mnist_training(nbla::Context ctx, string nnp_file_name) {

  // Create a context for cpu
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};

  // Create mnist data iterator
  MnistDataIterator data_iterator;

  // Get filename of the initialized model.
  const std::string nnp_file(nnp_file_name);

  // Create a Nnp object
  nbla::utils::nnp::Nnp nnp(ctx);

  // Set nnp file to Nnp object.
  if (!nnp.add(nnp_file)) {
    fprintf(stderr, "Error in loading nnp file.");
    return false;
  }

  // Get network.
  auto net = nnp.get_network("training");
  int batch_size = 128;
  net->set_batch_size(batch_size);

  // Get input data as a CPU array.
  nbla::CgVariablePtr x = net->get_variable("x");
  nbla::CgVariablePtr t = net->get_variable("t");
  nbla::CgVariablePtr loss = net->get_variable("loss");

  // Setup solver and input learnable parameters
  auto adam = create_AdamSolver(ctx, 0.001, 0.9, 0.999, 1.0e-8);
  adam->set_parameters(nnp.get_parameters());

  // Execute training
  FILE *fp;
  fp = fopen("log.txt", "wt");
  if (fp == NULL) {
    fprintf(stderr, "Error in opening log file.");
    return false;
  }
  int max_iter = 10000;
  int n_val_iter = 10;
  float mean_loss = 0.;
  for (int iter = 0; iter < max_iter; iter++) {

    // Get batch and copy to input variables
    const vector<Pair> batch = data_iterator.get_batch(batch_size);
    float_t *x_d = x->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx);
    uint8_t *t_d = t->variable()->cast_data_and_get_pointer<uint8_t>(cpu_ctx);

    // The following stride represents the tensor size less than the specified
    // index.
    // e.g. If the tensor size of x is (b, c, w, h) then stride_x becomes c * w
    // * h
    const int stride_x = x->variable()->strides()[0];
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < stride_x; j++)
        x_d[i * stride_x + j] = (float_t)(batch[i].data[j]);
      t_d[i] = batch[i].label;
    }

    // Execute forward, backward and update
    adam->zero_grad();
    loss->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/true);
    loss->variable()->grad()->fill(1.0);
    loss->backward(/*NdArrayPtr grad =*/nullptr, /*bool clear_buffer = */ true);
    adam->update();

    // Get and print the average loss
    float_t *loss_d =
        loss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx);
    mean_loss += loss_d[0];
    if ((iter + 1) % n_val_iter == 0) {
      mean_loss /= n_val_iter;
      fprintf(fp, "iter: %d, loss: %f\n", iter + 0, mean_loss);
      std::cout << "iter: " << iter + 0 << ", loss: " << mean_loss << std::endl;
      mean_loss = 0;
    }
  }
  fclose(fp);

  // Save parameters
  if (!nnp.save_parameters("parameter.protobuf")) {
    fprintf(stderr, "Error in saving parameters.");
    return false;
  }

  return true;
}
