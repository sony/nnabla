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
#include <random>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define open _open
#define O_RDONLY _O_RDONLY
#endif

using namespace std;
namespace nbla {
namespace utils {
namespace nnp {

const nbla::Context kCpuCtx{{"cpu:float"}, "CpuCachedArray", "0"};

DataIteratorFromCacheFiles::DataIteratorFromCacheFiles(
    shared_ptr<DatasetImpl> dataset)
    : n_data_(0), batch_size_(0), shuffle_(0), current_id_(0) {
  int n_stream = dataset->get_num_stream();
  int no_image_normalization_ = dataset->no_image_normalization();
  vector<Shape_t> shapes = dataset->get_shapes();
  vector<vector<NdArrayPtr>> cache_blocks = dataset->get_cache_blocks();

  n_data_ = dataset->get_num_data();
  batch_size_ = dataset->batch_size();
  shuffle_ = dataset->shuffle();
  data_names_ = dataset->get_data_names();

  dataset_.resize(n_stream);
  for (int i = 0; i < n_stream; i++) {

    Shape_t shape = shapes[i];
    shape[0] = n_data_;
    dataset_[i] = make_shared<NdArray>(shape);

    float normalize = 1.0;
    if (!no_image_normalization_ &&
        cache_blocks[0][i]->array()->dtype() == nbla::dtypes::UBYTE &&
        shape.size() > 2) {
      normalize = 1.0 / 255.0;
    }
    float *target =
        dataset_[i]->cast(get_dtype<float>(), kCpuCtx, true)->pointer<float>();
    for (int j = 0; j < cache_blocks.size(); j++) {
      NdArrayPtr ndarray = cache_blocks[j][i];
      float *source =
          ndarray->cast(get_dtype<float>(), kCpuCtx, false)->pointer<float>();
      for (int k = 0; k < ndarray->size(); k++)
        target[ndarray->size() * j + k] = source[k] * normalize;
    }
  }

  for (int i = 0; i < n_data_; i++)
    shuffle_ids_.push_back(i);
}

DataIteratorFromCacheFiles::~DataIteratorFromCacheFiles() {}

void DataIteratorFromCacheFiles::shuffle_index() {
  random_device rd;
  mt19937 g(rd());
  shuffle(std::begin(shuffle_ids_), std::end(shuffle_ids_), g);
}

void DataIteratorFromCacheFiles::shuffle_dataset() {

  if (!shuffle_)
    return;

  shuffle_index();
  vector<NdArrayPtr> source = dataset_;

  for (int i = 0; i < source.size(); i++) {
    NdArrayPtr sarray = source[i];
    NdArrayPtr tarray = make_shared<NdArray>(sarray->shape());
    float *x =
        sarray->cast(get_dtype<float>(), kCpuCtx, false)->pointer<float>();
    float *y =
        tarray->cast(get_dtype<float>(), kCpuCtx, true)->pointer<float>();
    const int dsize = sarray->strides()[0];
    for (int t = 0; t < n_data_; t++) {
      const int s = shuffle_ids_[t];
      memcpy(y + t * dsize, x + s * dsize, sizeof(float) * dsize);
    }
    x = sarray->cast(get_dtype<float>(), kCpuCtx, true)->pointer<float>();
    y = tarray->cast(get_dtype<float>(), kCpuCtx, false)->pointer<float>();
    memcpy(x, y, sizeof(float) * n_data_ * dsize);
  }
}

const vector<string> DataIteratorFromCacheFiles::get_data_names() const {
  return data_names_;
}

const int DataIteratorFromCacheFiles::get_batch_size() const {
  return batch_size_;
}

const int DataIteratorFromCacheFiles::get_iter_per_epoch() const {
  return int(n_data_ / batch_size_);
}

unordered_map<string, NdArrayPtr> DataIteratorFromCacheFiles::next() {

  const int n_stream = data_names_.size();
  unordered_map<string, NdArrayPtr> data_batch;

  // allocate batch
  for (int n = 0; n < n_stream; n++) {
    NdArrayPtr sarray = dataset_[n];
    Shape_t shape_x = sarray->shape();
    shape_x[0] = batch_size_;
    NdArrayPtr tarray = make_shared<NdArray>(shape_x);
    data_batch.insert({data_names_[n], tarray});
  }

  // before shuffle
  int n_count = batch_size_;
  if (n_data_ <= current_id_ + batch_size_)
    n_count = n_data_ - current_id_;
  for (int n = 0; n < n_stream; n++) {
    NdArrayPtr sarray = dataset_[n];
    NdArrayPtr tarray = data_batch.at(data_names_[n]);
    float *source =
        sarray->cast(get_dtype<float>(), kCpuCtx, false)->pointer<float>();
    float *target =
        tarray->cast(get_dtype<float>(), kCpuCtx, true)->pointer<float>();
    const int size_x = sarray->strides()[0];
    source += size_x * current_id_;
    memcpy(target, source, sizeof(float) * size_x * n_count);
  }
  current_id_ += n_count;

  // shuffle
  if (n_data_ <= current_id_) {
    shuffle_dataset();
    current_id_ = 0;
  }

  // after shuffle
  for (int n = 0; n < n_stream; n++) {
    NdArrayPtr sarray = dataset_[n];
    NdArrayPtr tarray = data_batch.at(data_names_[n]);
    float *source =
        sarray->cast(get_dtype<float>(), kCpuCtx, false)->pointer<float>();
    float *target =
        tarray->cast(get_dtype<float>(), kCpuCtx, true)->pointer<float>();
    const int size_x = sarray->strides()[0];
    source += size_x * current_id_;
    target += size_x * n_count;
    memcpy(target, source, sizeof(float) * size_x * (batch_size_ - n_count));
  }
  current_id_ += batch_size_ - n_count;

  return data_batch;
}
}
}
}
