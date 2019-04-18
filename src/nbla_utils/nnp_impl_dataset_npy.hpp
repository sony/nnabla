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
};

// ----------------------------------------------------------------------
// NpyReader
// ----------------------------------------------------------------------
/** Helper class which represent a npy file.
*/
class NpyReader {
private:
  std::string filename_;
  shared_ptr<FileResource> file_;
  int num_of_data_;
  int major_version_;
  int minor_version_;
  bool fortran_order_;
  Shape_t shape_;
  int word_size_;

public:
  NpyReader(const std::string &filename);
  ~NpyReader();

  bool preload(vector<Shape_t> &shapes);
  bool load_all(vector<NdArrayPtr> &cached);
  int num_of_data() const { return num_of_data_; };
  const string &get_name() const { return filename_; }
};

// ----------------------------------------------------------------------
// DatasetNpyImpl
// ----------------------------------------------------------------------
/** Implementation of DatasetNpyImpl
*/
class DatasetNpyImpl : public DatasetImpl {
private:
  // Number of data
  int n_data_;

  // Number of stream
  int n_stream_;

  // Data shapes
  std::vector<Shape_t> shapes_;

  // Data names
  std::vector<string> data_names_;

  // Cache blocks
  std::vector<vector<NdArrayPtr>> cache_blocks_; // cache_block/data_column/data
  std::vector<std::shared_ptr<NpyReader>> cached_variables_;

public:
  DatasetNpyImpl(const ::Dataset &dataset);
  virtual const int get_num_stream() const override;
  virtual const int get_num_data() const override;
  virtual std::vector<string> get_data_names() override;
  virtual std::vector<Shape_t> get_shapes() override;
  virtual std::vector<vector<NdArrayPtr>> get_cache_blocks() override;
};

} // namespace nnp
} // namespace utils
} // namespace nbla

#endif
