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
// Include nnabla header files

#include <fstream>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>
#include <zlib.h>
using namespace std;

#include <nbla/computation_graph/variable.hpp>
#include <nbla/context.hpp>
using std::make_shared;

/******************************************/
// Example of data provider
/******************************************/

vector<vector<uint8_t>> read_images(const string &train,
                                    const string &data_root,
                                    const string &filename) {

  gzFile fp = gzopen((data_root + filename).c_str(), "rb");
  if (fp == NULL) {
    cerr << "This sample requires mnist data downloaded before." << endl;
    exit(0);
  }

  char header[16];
  gzread(fp, header, 16);
  int num_images = 60000;
  if (train == "test")
    num_images = 10000;

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
  gzclose(fp);
  return images;
}

vector<uint8_t> read_labels(const string &train, const string &data_root,
                            const string &filename) {

  gzFile fp = gzopen((data_root + filename).c_str(), "rb");
  if (fp == NULL) {
    cerr << "This sample requires mnist data downloaded before." << endl;
    exit(0);
  }

  char header[8];
  gzread(fp, header, 8);
  int num_images = 60000;
  if (train == "test")
    num_images = 10000;

  vector<uint8_t> labels;
  for (int i = 0; i < num_images; ++i) {
    unsigned char temp = 0;
    gzread(fp, (char *)&temp, sizeof(temp));
    labels.push_back(temp);
  }
  gzclose(fp);
  return labels;
}

struct Pair {
  std::vector<uint8_t> data;
  int label;
};

class MnistDataIterator {
private:
public:
  MnistDataIterator(string train);
  ~MnistDataIterator();

  vector<Pair> dataset_;
  int current_idx_;
  string train_;
  void permutation();
  const vector<Pair> get_batch(int batchsize);
  void provide_data(Context ctx, int batch_size, CgVariablePtr x,
                    CgVariablePtr t);
};

MnistDataIterator::MnistDataIterator(string train)
    : current_idx_(0), train_(train) {
  string image_gz = "train-images-idx3-ubyte.gz";
  string label_gz = "train-labels-idx1-ubyte.gz";
  if (train == "test") {
    image_gz = "t10k-images-idx3-ubyte.gz";
    label_gz = "t10k-labels-idx1-ubyte.gz";
  }
  vector<vector<uint8_t>> images = read_images(train, "./", image_gz);
  vector<uint8_t> labels = read_labels(train, "./", label_gz);
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

void set_data(Context cpu_ctx, vector<Pair> batch_data, CgVariablePtr x,
              CgVariablePtr t) {

  int batch_size = batch_data.size();

  float_t *x_d =
      x->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, true);
  const int stride_x = x->variable()->strides()[0];
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < stride_x; j++)
      x_d[i * stride_x + j] = (float_t)(batch_data[i].data[j]) / 255.;
  }

  uint8_t *t_d =
      t->variable()->cast_data_and_get_pointer<uint8_t>(cpu_ctx, true);
  for (int i = 0; i < batch_size; i++)
    t_d[i] = batch_data[i].label;
}

void MnistDataIterator::provide_data(Context cpu_ctx, int batch_size,
                                     CgVariablePtr x, CgVariablePtr t) {

  auto batch_data = get_batch(batch_size);
  set_data(cpu_ctx, get_batch(batch_size), x, t);
}
