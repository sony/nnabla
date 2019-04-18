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

static const int MAX_NAME = 1024;
const nbla::Context kCpuCtx{{"cpu:float"}, "CpuCachedArray", "0"};

#ifdef NBLA_UTILS_WITH_HDF5
#include <hdf5.h>
#include <hdf5_hl.h>

bool read_dataset(hid_t gid, const int id, string &data_name,
                  NdArrayPtr ndarray) {
  char name[MAX_NAME];

  if (H5Gget_objname_by_idx(gid, (hsize_t)id, name, (size_t)MAX_NAME) < 0)
    return false;
  data_name = string(name);

  hid_t did = H5Dopen(gid, name, H5P_DEFAULT);
  hid_t sp = H5Dget_space(did);
  int rank = H5Sget_simple_extent_ndims(sp);
  hsize_t dims[rank];
  if (!H5Sget_simple_extent_dims(sp, dims, NULL))
    return false;

  // NdArray ndarray;;
  Shape_t shape_x(rank);
  for (int i = 0; i < rank; i++)
    shape_x[i] = dims[i];
  ndarray->reshape(shape_x, true);

  hid_t tid = H5Dget_type(did);

  if (H5Tequal(tid, H5T_NATIVE_CHAR)) {
    char *buffer =
        ndarray->cast(nbla::get_dtype<char>(), kCpuCtx, true)->pointer<char>();
    H5Dread(did, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  }
  if (H5Tequal(tid, H5T_NATIVE_UCHAR)) {
    uint8_t *buffer = ndarray->cast(nbla::get_dtype<uint8_t>(), kCpuCtx, true)
                          ->pointer<uint8_t>();
    H5Dread(did, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  }
  if (H5Tequal(tid, H5T_NATIVE_SHORT)) {
    short *buffer = ndarray->cast(nbla::get_dtype<short>(), kCpuCtx, true)
                        ->pointer<short>();
    H5Dread(did, H5T_NATIVE_SHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  }
  if (H5Tequal(tid, H5T_NATIVE_INT)) {
    int *buffer =
        ndarray->cast(nbla::get_dtype<int>(), kCpuCtx, true)->pointer<int>();
    H5Dread(did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  }
  if (H5Tequal(tid, H5T_NATIVE_UINT)) {
    unsigned int *buffer =
        ndarray->cast(nbla::get_dtype<unsigned int>(), kCpuCtx, true)
            ->pointer<unsigned int>();
    H5Dread(did, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  }
  if (H5Tequal(tid, H5T_NATIVE_LONG)) {
    long *buffer =
        ndarray->cast(nbla::get_dtype<long>(), kCpuCtx, true)->pointer<long>();
    H5Dread(did, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  }
  if (H5Tequal(tid, H5T_NATIVE_ULONG)) {
    unsigned long *buffer =
        ndarray->cast(nbla::get_dtype<unsigned long>(), kCpuCtx, true)
            ->pointer<unsigned long>();
    H5Dread(did, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  }
  if (H5Tequal(tid, H5T_NATIVE_LLONG)) {
    long long *buffer =
        ndarray->cast(nbla::get_dtype<long long>(), kCpuCtx, true)
            ->pointer<long long>();
    H5Dread(did, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  }
  if (H5Tequal(tid, H5T_NATIVE_FLOAT)) {
    float *buffer = ndarray->cast(nbla::get_dtype<float>(), kCpuCtx, true)
                        ->pointer<float>();
    H5Dread(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  }
  if (H5Tequal(tid, H5T_NATIVE_DOUBLE)) {
    double *buffer = ndarray->cast(nbla::get_dtype<double>(), kCpuCtx, true)
                         ->pointer<double>();
    H5Dread(did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
  }
  H5Dclose(did);
  return true;
}

bool parse_hdf5_group(hid_t gid, vector<string> &data_names,
                      vector<NdArrayPtr> &ndarrays) {

  hsize_t num = 0;
  if (H5Gget_num_objs(gid, &num) < 0)
    return false;

  char group_name[MAX_NAME];
  if (H5Iget_name(gid, group_name, MAX_NAME) < 0)
    return false;

  char name[MAX_NAME];

  vector<int> dataset_ids;
  for (int i = 0; i < num; i++) {
    if (H5Gget_objname_by_idx(gid, (hsize_t)i, name, (size_t)MAX_NAME) < 0)
      return false;
    if (H5Gget_objtype_by_idx(gid, i) == H5G_DATASET)
      dataset_ids.push_back(i);
  }

  const int n_stream = dataset_ids.size();
  data_names.resize(n_stream);
  for (int n = 0; n < n_stream; n++) {
    ndarrays.push_back(make_shared<NdArray>());
  }

  for (int i = 0; i < dataset_ids.size(); i++) {
    read_dataset(gid, dataset_ids[i], data_names[i], ndarrays[i]);
  }

  return true;
}

bool read_file_to_str(const std::string &filename, std::string &data) {

  std::ifstream file(filename.c_str(), std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Error: could not open file " << filename << std::endl;
    return false;
  }

  size_t filesize = static_cast<size_t>(file.tellg());
  if (filesize <= 0) {
    file.close();
    std::cerr << "Error: file size 0" << std::endl;
    return false;
  }

  data.resize(filesize);
  file.seekg(0, file.beg);
  file.read(&data.front(), filesize);
  file.close();

  return true;
}
#endif

bool load_cache(const string &filename, vector<string> &data_names,
                vector<NdArrayPtr> &ndarrays) {
#if NBLA_UTILS_WITH_HDF5
  string h5data;
  if (!read_file_to_str(filename, h5data)) {
    return false;
  }

  hid_t id = H5LTopen_file_image((char *)&h5data.front(), h5data.size(),
                                 H5LT_FILE_IMAGE_DONT_RELEASE);
  if (0 <= id) {
    hid_t root = H5Gopen(id, "/", H5P_DEFAULT);
    if (0 <= root)
      parse_hdf5_group(root, data_names, ndarrays);
  }
  return true;
#else
  std::cerr << "Error: use -DNNABLA_UTILS_WITH_HDF5 options in cmake."
            << std::endl;
  return false;
#endif
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
}
}
}
