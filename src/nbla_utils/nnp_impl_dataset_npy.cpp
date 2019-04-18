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

#include "nnp_impl.hpp"
#include "nnp_impl_dataset_npy.hpp"
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

#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <typeinfo>
#include <vector>

using namespace std;
namespace nbla {
namespace utils {
namespace nnp {

static const int BUF_SZ = 256;
static const char npy_head_magic_num = 0x93;
static const size_t HEAD_VER_1_0 = 13;
const nbla::Context kCpuCtx{{"cpu:float"}, "CpuCachedArray", "0"};

// ----------------------------------------------------------------------
// Helper functions
// ----------------------------------------------------------------------
bool has_suffix(const string &s, const string &suffix) {
  return (s.size() >= suffix.size()) &&
         equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

// We supposed the variable name can be obtained from
// filename, as the following:
//   input_image.npy
//      --> the variable name is : input_image
string get_varname(const string &s) {
  string filename(s);
  size_t last_index = filename.rfind('.');
  return filename.substr(0, last_index);
}

bool search_for_cache_files(const string &path,
                            vector<shared_ptr<NpyReader>> &cache_files,
                            vector<string> &data_names) {
#ifndef _WIN32
  DIR *dir = opendir(path.c_str());

  if (!dir) {
    std::cerr << "Could not find directory " << path << std::endl;
    closedir(dir);
    return false;
  }

  dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (has_suffix(entry->d_name, ".npy")) {
      string d_name(entry->d_name);
      string varname = get_varname(entry->d_name);
      string filename = path + '/' + d_name;
      cache_files.push_back(make_shared<NpyReader>(filename));
      data_names.push_back(varname);
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
// FileResource
// ----------------------------------------------------------------------
FileResource::FileResource(const std::string &filename)
    : fp_(fopen(filename.c_str(), "rb")) {}

FileResource::~FileResource() {
  if (fp_ != nullptr) {
    fclose(fp_);
  }
}

bool FileResource::read(char *buffer, size_t size) {
  if (fread(buffer, sizeof(char), size, fp_) != size) {
    return false;
  }
  return true;
}

bool FileResource::gets(char *buffer, size_t size) {
  if (fgets(buffer, size, fp_) == nullptr) {
    return false;
  }
  return true;
}

void FileResource::seek_to(size_t absolute_offset) {
  fseek(fp_, absolute_offset, SEEK_SET);
}

// ----------------------------------------------------------------------
// NpyReader
// ----------------------------------------------------------------------
NpyReader::NpyReader(const string &filename)
    : filename_(filename), file_(make_shared<FileResource>(filename)),
      num_of_data_(0), major_version_(0), minor_version_(0),
      fortran_order_(false), shape_(), word_size_(0) {}

NpyReader::~NpyReader() {}

bool NpyReader::preload(vector<Shape_t> &shapes) {
  vector<char> buffer(BUF_SZ);
  size_t pos = 0;

  if (!file_->read(buffer.data(), HEAD_VER_1_0))
    return false;

  if (buffer[0] != npy_head_magic_num) {
    return false;
  }

  string magic("NUMPY");

  if (!equal(magic.begin(), magic.end(), buffer.begin() + 1)) {
    return false;
  }

  pos += magic.size() + sizeof(char);

  major_version_ = *reinterpret_cast<char *>(buffer.data() + (pos++));
  minor_version_ = *reinterpret_cast<char *>(buffer.data() + (pos++));

  uint16_t header_len = *reinterpret_cast<uint16_t *>(buffer.data() + pos);

  if ((major_version_ == 2) || (header_len >= (uint16_t)65535)) {
    pos += sizeof(uint32_t);
  } else {
    pos += sizeof(uint16_t);
  }

  file_->seek_to(pos);
  file_->gets(buffer.data(), BUF_SZ);

  string header(buffer.data(), header_len);
  if (header[header.size() - 1] != '\n') {
    return false;
  }

  regex e("\'(\\w*)\': (\\S*)");
  smatch m;
  string s = header;
  while (regex_search(s, m, e)) {
    if (m.size() != 3) {
      return false;
    }

    if (m[1] == "fortran_order") {
      fortran_order_ = (m[2] == "True" ? true : false);
      regex ex("[0-9][0-9]*");
      smatch ms;
      string ss = m.suffix().str();
      while (regex_search(ss, ms, ex)) {
        shape_.push_back(stoi(ms[0].str()));
        ss = ms.suffix().str();
      }
      break;
    } else if (m[1] == "descr") {
      string descr = m[2].str();
      bool little_endian = (descr[1] == '<' || descr[1] == '|' ? true : false);
      if (!little_endian) {
        return false;
      }

      if (descr[2] != 'f') {
        // FIXME: Unnecessary limitation.
        // This is a limitation introduced by
        // data_iterator's implementation.
        cerr << "numpy array's data type is not float32." << endl;
        return false;
      }

      word_size_ = stoi(descr.substr(3));
      if (word_size_ != 4) {
        // FIXME: Unnecessary limitation.
        // This is a limitation introduced by
        // data_iterator's implementation.
        cerr << "numpy array's data type is not float32." << endl;
        return false;
      }
    }

    s = m.suffix().str();
  }

  num_of_data_ = shape_[0];
  shapes.push_back(shape_);
  return true;
}

bool NpyReader::load_all(vector<NdArrayPtr> &cached) {

  NdArrayPtr ndarray = make_shared<NdArray>(shape_);
  float *buffer =
      ndarray->cast(nbla::get_dtype<float>(), kCpuCtx)->pointer<float>();
  int read_bytes = ndarray->size() * word_size_;
  file_->read(reinterpret_cast<char *>(buffer), read_bytes);
  cached.push_back(ndarray);
  return true;
}

// ----------------------------------------------------------------------
// DatasetNpyImpl
// ----------------------------------------------------------------------
DatasetNpyImpl::DatasetNpyImpl(const ::Dataset &dataset)
    : DatasetImpl(dataset), n_data_(0), n_stream_(0), shapes_(0),
      data_names_(0), cache_blocks_(0) {

  vector<NdArrayPtr> cached;
  search_for_cache_files(this->cache_dir(), cached_variables_, data_names_);
  for (auto v : cached_variables_) {
    if (!v->preload(shapes_)) {
      std::cerr << "Could not preload cache file: " << v->get_name()
                << std::endl;
    }

    if (!v->load_all(cached)) {
      std::cerr << "Could not load all cache file: " << v->get_name()
                << std::endl;
    }
  }

  cache_blocks_.push_back(cached);

  n_stream_ = cached_variables_.size();
  if (n_stream_ <= 0) {
    std::cerr << "Could not find any cached variable." << endl;
    return;
  }
  n_data_ = cached_variables_[0]->num_of_data();
}

const int DatasetNpyImpl::get_num_stream() const { return n_stream_; }
const int DatasetNpyImpl::get_num_data() const { return n_data_; }
vector<string> DatasetNpyImpl::get_data_names() { return data_names_; }
vector<Shape_t> DatasetNpyImpl::get_shapes() { return shapes_; }
vector<vector<NdArrayPtr>> DatasetNpyImpl::get_cache_blocks() {
  return cache_blocks_;
}

} // namespace nnp
} // namespace utils
} // namespace nbla