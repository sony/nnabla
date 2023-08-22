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

#include "hdf5_wrapper.hpp"
#include "nnp_impl.hpp"
#include <fstream>
#include <iostream>

#ifndef _WIN32
#include <dirent.h>
#endif

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

bool load_cache(const string &filename, vector<string> &data_names,
                vector<NdArrayPtr> &ndarrays) {
  return load_from_h5_file(filename, data_names, ndarrays);
}

bool has_suffix(const string &s, const string &suffix) {
  return (s.size() >= suffix.size()) &&
         equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

bool create_cachefile_list(const string &path, vector<string> &filenames) {

#ifndef _WIN32
  DIR *dir = opendir(path.c_str());

  if (!dir) {
    std::cerr << "Could not find directory " << path << std::endl;
    closedir(dir);
    return 1;
  }

  dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (has_suffix(entry->d_name, ".h5")) {
      string d_name(entry->d_name);
      filenames.push_back(path + '/' + d_name);
    }
  }
  closedir(dir);

  return true;
#else
  std::cerr << "Error: Windows OS are not supported yet." << std::endl;
  return false;
#endif
}

// ----------------------------------------------------------------------
// DatasetHDF5Impl
// ----------------------------------------------------------------------
DatasetHDF5Impl::DatasetHDF5Impl(const ::Dataset &dataset)
    : DatasetImpl(dataset), n_data_(0), n_stream_(0) {

  vector<string> filenames;
  create_cachefile_list(this->cache_dir(), filenames);

  shapes_.clear();
  n_data_ = 0;
  for (auto filename : filenames) {
    vector<NdArrayPtr> ndarrays;
    if (!load_cache(filename, data_names_, ndarrays))
      std::cerr << "Error: file load." << std::endl;
    n_stream_ = ndarrays.size();
    for (auto ndarray : ndarrays)
      shapes_.push_back(ndarray->shape());
    n_data_ += shapes_[0][0];
    cache_blocks_.push_back(ndarrays);
  }
}

const int DatasetHDF5Impl::get_num_stream() const { return n_stream_; }
const int DatasetHDF5Impl::get_num_data() const { return n_data_; }
vector<string> DatasetHDF5Impl::get_data_names() { return data_names_; }
vector<Shape_t> DatasetHDF5Impl::get_shapes() { return shapes_; }
vector<vector<NdArrayPtr>> DatasetHDF5Impl::get_cache_blocks() {
  return cache_blocks_;
}
} // namespace nnp
} // namespace utils
} // namespace nbla
