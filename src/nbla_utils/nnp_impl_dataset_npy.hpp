// Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

#ifndef NBLA_UTILS_NNP_IMPL_DATASET_NPY_HPP_
#define NBLA_UTILS_NNP_IMPL_DATASET_NPY_HPP_

#include "nnp_impl.hpp"
#include <iostream>
#include <queue>
#include <string>

#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/solver.hpp>
#include <nbla_utils/nnp.hpp>

namespace nbla {
namespace utils {
namespace nnp {

// ----------------------------------------------------------------------
// FileResource
// ----------------------------------------------------------------------
/** Helper class FileResource which RAII to avoid using exception as fstream.
*/
struct FileResource {
  FILE *fp_;
  FileResource(const std::string &filename);
  ~FileResource();
  bool read(char *buffer, size_t size);
  bool gets(char *buffer, size_t size);
  void seek_to(size_t absolute_offset);
  void seek_by(size_t relative_offset);
  size_t current_position() const;
};

struct VariableDesc {
  VariableDesc();
  dtypes data_type;
  int word_size;
  Shape_t shape;
  size_t offset;
  bool valid;
};

// ----------------------------------------------------------------------
// CacheFile
// ----------------------------------------------------------------------
/** Corresponding to each *.npy cache file.
*/
class CacheFile {
public:
  CacheFile(const string &filename, vector<string> v_names);
  virtual ~CacheFile();

  void preload();
  bool read_data(string v_name, void *buffer);
  const VariableDesc *get_variable_desc(string v_name);
  int get_num_data() const;
  const string get_name() const;

private:
  bool load_variable(string v_name, int &offset);

private:
  string filename_;
  shared_ptr<FileResource> file_;
  int num_of_data_;
  int major_version_;
  int minor_version_;
  bool fortran_order_;
  vector<string> v_names_;
  typedef unordered_map<string, shared_ptr<VariableDesc>> var_desc_map_t;
  var_desc_map_t var_desc_;
};

// ----------------------------------------------------------------------
// RingBuffer
// ----------------------------------------------------------------------
/** Make the balance between memory usage and performance
*/
class RingBuffer {
public:
  RingBuffer(const vector<shared_ptr<CacheFile>> &cache_files, int batch_size,
             string variable_name, const vector<int> &idx_list, bool shuffle);

  virtual ~RingBuffer();

  void fill_up();
  void read_batch_data(int idx, shared_ptr<VariableBuffer> v);
  Shape_t get_shape() const;
  dtypes get_data_type() const;

  RingBuffer(const RingBuffer &) = delete;
  RingBuffer &operator=(const RingBuffer &) = delete;

private:
  void fill_buffer(int load_size);
  vector<shared_ptr<CacheFile>> cache_files_;
  vector<int> idx_list_;
  bool shuffle_;
  int prev_index_;
  int current_;
  int start_;
  int total_;
  int total_buffer_size_;
  int cache_file_data_size_;
  int data_size_;
  int batch_data_size_;
  char *buffer_;
  char *shuffle_buffer_;
  std::queue<CacheFile *> file_queue_;
  dtypes data_type_;
  Shape_t shape_;
  string variable_name_;
};

// ----------------------------------------------------------------------
// DatasetNpyCache
// ----------------------------------------------------------------------
/** Implementation of DatasetNpyCache
*/
class DatasetNpyCache : public DatasetImpl {
  friend class NnpImpl;

public:
  DatasetNpyCache(const ::Dataset &dataset);
  virtual ~DatasetNpyCache() = default;
  virtual unordered_map<string, shared_ptr<VariableBuffer>>
  get_batch_data(uint32_t iter_num) override;
  virtual int get_num_data() const override;
  virtual vector<string> get_data_names() override;

private:
  vector<string> data_names_;
  int num_of_data_;
  unordered_map<string, shared_ptr<RingBuffer>> ring_buffers_;
};

} // namespace nnp
} // namespace utils
} // namespace nbla

#endif
