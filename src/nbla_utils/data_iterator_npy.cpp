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
#include <thread>

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
    : dataset_(dataset), workers_(), iter_(0), req_exit_(false), queue_() {
  int worker_num = 1;
  for (int i = 0; i < worker_num; ++i) {
    workers_.push_back(make_shared<thread>(
        [&](DataIteratorFromCacheFiles *difc) { difc->loop(); }, this));
  }
}

DataIteratorFromCacheFiles::~DataIteratorFromCacheFiles() {
  req_exit_ = true;
  for (auto &w : workers_) {
    w->join();
  }
}

void DataIteratorFromCacheFiles::loop() {
  while (!req_exit_) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto d = dataset_->get_batch_data(iter_++);
    while (queue_.size() >= MAX_ITEM) {
      full_cond_.wait_for(lock, std::chrono::milliseconds(TIMEOUT));
      if (req_exit_)
        return;
    }
    queue_.push(d);
    empty_cond_.notify_one();
  }
}

const vector<string> DataIteratorFromCacheFiles::get_data_names() const {
  return dataset_->get_data_names();
}

const int DataIteratorFromCacheFiles::get_batch_size() const {
  return dataset_->batch_size();
}

const int DataIteratorFromCacheFiles::get_iter_per_epoch() const {
  return int(dataset_->get_num_data() / dataset_->batch_size());
}

unordered_map<string, NdArrayPtr> DataIteratorFromCacheFiles::next() {
  std::unique_lock<std::mutex> lock(mutex_);
  unordered_map<string, shared_ptr<VariableBuffer>> d;
  unordered_map<string, NdArrayPtr> x;
  while (queue_.empty()) {
    empty_cond_.wait_for(lock, std::chrono::milliseconds(TIMEOUT));
    if (req_exit_) {
      cout << "exit dequeue" << endl;
      d = dataset_->get_batch_data(iter_++);
      break;
    }
  }
  d = queue_.front();
  queue_.pop();
  full_cond_.notify_one();
  for (auto kv : d) {
    x[kv.first] = kv.second->to_ndarray();
  }
  return x;
}
}
}
}
