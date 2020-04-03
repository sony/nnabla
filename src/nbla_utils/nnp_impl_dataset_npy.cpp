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

#include "nnabla.pb.h"
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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
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

// buffer length = 5 * cache_file_size
const int NUM_OF_CACHE_FILE = 5;

const nbla::Context kCpuCtx{{"cpu:float"}, "CpuCachedArray", "0"};

// ----------------------------------------------------------------------
// Helper functions
// ----------------------------------------------------------------------
bool load_variable_list(const string &path, vector<string> &data_names) {
  ifstream cache_info_file(path + "/cache_info.csv");
  std::string variable_name;
  while (std::getline(cache_info_file, variable_name)) {
    data_names.push_back(variable_name);
  }
  return true;
}

bool has_suffix(const string &s, const string &suffix) {
  return (s.size() >= suffix.size()) &&
         equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

bool search_for_cache_files(const string &path,
                            const vector<string> &data_names,
                            vector<shared_ptr<CacheFile>> &cache_files) {

  ifstream cache_index_info(path + "/cache_index.csv");
  std::string line;
  while (std::getline(cache_index_info, line)) {
    string filename = path + "/" + line.substr(0, line.find(","));
    cache_files.push_back(make_shared<CacheFile>(filename, data_names));
  }
  return true;
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
  assert(fp_ != nullptr);
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

void FileResource::seek_by(size_t relative_offset) {
  fseek(fp_, relative_offset, SEEK_CUR);
}

size_t FileResource::current_position() const { return ftell(fp_); }

// ----------------------------------------------------------------------
// VariableDesc
// ----------------------------------------------------------------------
VariableDesc::VariableDesc()
    : data_type(dtypes::FLOAT), word_size(4), shape(), offset(0), valid(false) {
}

// ----------------------------------------------------------------------
// VariableBuffer
// ----------------------------------------------------------------------
VariableBuffer::VariableBuffer()
    : buffer_(nullptr), data_type_(dtypes::UBYTE), block_size_(0), shape_() {}

VariableBuffer::~VariableBuffer() {
  if (buffer_) {
    delete[] buffer_;
  }
}

void VariableBuffer::from_buffer(const void *buffer, dtypes data_type,
                                 int block_size, Shape_t shape) {
  if (buffer_ == nullptr) {
    buffer_ = new char[block_size];
  }

  data_type_ = data_type;
  block_size_ = block_size;
  shape_ = shape;
  memcpy(buffer_, buffer, block_size);
}

NdArrayPtr VariableBuffer::to_ndarray() {
  NdArrayPtr v = make_shared<NdArray>(shape_);
  void *buffer = nullptr;

  switch (data_type_) {
  case dtypes::BYTE:
    buffer = v->cast(nbla::get_dtype<char>(), kCpuCtx, true)->pointer<char>();
    break;
  case dtypes::UBYTE:
    buffer =
        v->cast(nbla::get_dtype<uint8_t>(), kCpuCtx, true)->pointer<uint8_t>();
    break;
  case dtypes::INT:
    buffer = v->cast(nbla::get_dtype<int>(), kCpuCtx, true)->pointer<int>();
    break;
  case dtypes::UINT:
    buffer = v->cast(nbla::get_dtype<unsigned int>(), kCpuCtx, true)
                 ->pointer<unsigned int>();
    break;
  case dtypes::LONG:
    buffer = v->cast(nbla::get_dtype<long>(), kCpuCtx, true)->pointer<long>();
    break;
  case dtypes::ULONG:
    buffer = v->cast(nbla::get_dtype<unsigned long>(), kCpuCtx, true)
                 ->pointer<unsigned long>();
    break;
  case dtypes::LONGLONG:
    buffer = v->cast(nbla::get_dtype<long long>(), kCpuCtx, true)
                 ->pointer<long long>();
    break;
  case dtypes::ULONGLONG:
    buffer = v->cast(nbla::get_dtype<unsigned long long>(), kCpuCtx, true)
                 ->pointer<unsigned long long>();
    break;
  case dtypes::FLOAT:
    buffer = v->cast(nbla::get_dtype<float>(), kCpuCtx, true)->pointer<float>();
    break;
  case dtypes::DOUBLE:
    buffer =
        v->cast(nbla::get_dtype<double>(), kCpuCtx, true)->pointer<double>();
    break;
  default:
    assert(false);
    break;
  }

  memcpy(buffer, buffer_, block_size_);
  return v;
}

// ----------------------------------------------------------------------
// CacheFile
// ----------------------------------------------------------------------
CacheFile::CacheFile(const string &filename, vector<string> v_names)
    : filename_(filename), file_(make_shared<FileResource>(filename)),
      num_of_data_(0), major_version_(0), minor_version_(0),
      fortran_order_(false), v_names_(v_names), var_desc_() {

  for (auto v : v_names_) {
    var_desc_[v] = make_shared<VariableDesc>();
  }
}

CacheFile::~CacheFile() {}

void CacheFile::preload(void) {
  int offset = 0;
  for (auto v : v_names_) {
    if (!load_variable(v, offset)) {
      cout << "failed to load:" << filename_ << endl;
    }
  }
}

bool CacheFile::load_variable(string v_name, int &offset) {
  Shape_t shape;
  vector<char> buffer(BUF_SZ);
  size_t pos = 0;

  auto var_desc = var_desc_[v_name];

  if (var_desc->valid) {
    return true;
  }

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

  file_->seek_to(pos + offset);
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
        shape.push_back(stoi(ms[0].str()));
        ss = ms.suffix().str();
      }
      break;
    } else if (m[1] == "descr") {
      string descr = m[2].str();
      bool little_endian = (descr[1] == '<' || descr[1] == '|' ? true : false);
      if (!little_endian) {
        return false;
      }

      switch (descr[2]) {
      case 'b':
        var_desc->word_size = 1;
        var_desc->data_type = dtypes::BYTE;
        break;
      case 'B':
        var_desc->word_size = 1;
        var_desc->data_type = dtypes::UBYTE;
        break;
      case 'i':
        var_desc->word_size = stoi(descr.substr(3));
        switch (var_desc->word_size) {
        case 1:
          var_desc->data_type = dtypes::BYTE;
          break;
        case 4:
          var_desc->data_type = dtypes::INT;
          break;
        case 8:
          var_desc->data_type = dtypes::LONGLONG;
        }
        break;
      case 'u':
        var_desc->word_size = stoi(descr.substr(3));
        switch (var_desc->word_size) {
        case 1:
          var_desc->data_type = dtypes::UBYTE;
          break;
        case 4:
          var_desc->data_type = dtypes::UINT;
          break;
        case 8:
          var_desc->data_type = dtypes::ULONGLONG;
        }
        break;
      case 'f':
        var_desc->word_size = stoi(descr.substr(3));
        if (var_desc->word_size == 4)
          var_desc->data_type = dtypes::FLOAT;
        else if (var_desc->word_size == 8)
          var_desc->data_type = dtypes::DOUBLE;
        break;
      }
    }

    s = m.suffix().str();
  }

  num_of_data_ = shape[0];
  var_desc->shape = shape;

  var_desc->offset = file_->current_position();
  var_desc->valid = true;
  file_->seek_by(compute_size_by_shape(shape) * var_desc->word_size);
  offset = file_->current_position();

  return true;
}

bool CacheFile::read_data(string v_name, void *buffer) {
  auto var_desc = var_desc_[v_name];
  int read_bytes = compute_size_by_shape(var_desc->shape) * var_desc->word_size;
  file_->seek_to(var_desc->offset);
  file_->read(reinterpret_cast<char *>(buffer), read_bytes);
  return true;
}

const VariableDesc *CacheFile::get_variable_desc(string v_name) {
  auto d = var_desc_[v_name];
  if (!d->valid) {
    return nullptr;
  }
  return d.get();
}

int CacheFile::get_num_data() const { return num_of_data_; }

const string CacheFile::get_name() const { return filename_; }

// class RingBuffer
RingBuffer::RingBuffer(const vector<shared_ptr<CacheFile>> &cache_files,
                       int batch_size, string variable_name,
                       const vector<int> &idx_list, bool shuffle)
    : cache_files_(cache_files), idx_list_(idx_list), shuffle_(shuffle),
      prev_index_(0), current_(0), start_(0), total_(0), total_buffer_size_(0),
      cache_file_data_size_(0), data_size_(0), batch_data_size_(0),
      buffer_(nullptr), shuffle_buffer_(0), file_queue_(),
      data_type_(dtypes::FLOAT), shape_(), variable_name_(variable_name) {

  for (auto f : cache_files_) {
    file_queue_.push(f.get());
  }

  auto cache_file = cache_files[0];
  auto var_desc = cache_file->get_variable_desc(variable_name);
  int word_size = var_desc->word_size;
  data_type_ = var_desc->data_type;
  shape_ = var_desc->shape;
  shape_[0] = 1;
  data_size_ = compute_size_by_shape(shape_) * word_size;
  cache_file_data_size_ = data_size_ * cache_file->get_num_data();
  shape_[0] = batch_size;
  batch_data_size_ = compute_size_by_shape(shape_) * word_size;
  total_buffer_size_ = cache_file_data_size_ * NUM_OF_CACHE_FILE;

  buffer_ = new char[total_buffer_size_];
  shuffle_buffer_ = new char[cache_file_data_size_];
  fill_buffer(0);
}

RingBuffer::~RingBuffer() {
  delete[] buffer_;
  delete[] shuffle_buffer_;
}

void RingBuffer::fill_buffer(int load_size) {
  while (load_size <= total_buffer_size_ - cache_file_data_size_) {
    if (file_queue_.empty()) {
      for (auto f : cache_files_) {
        file_queue_.push(f.get());
      }
    }
    auto file = file_queue_.front();
    if (shuffle_) {
      char *dest = buffer_ + load_size;
      file->read_data(variable_name_, shuffle_buffer_);
      for (int t = 0; t < idx_list_.size(); ++t) {
        const int s = idx_list_[t];
        const int d = data_size_;
        memcpy(dest + t * d, shuffle_buffer_ + s * d, d);
      }
    } else {
      file->read_data(variable_name_, buffer_ + load_size);
    }
    file_queue_.pop();
    load_size += cache_file_data_size_;
  }
  total_ = load_size;
}

void RingBuffer::fill_up() {
  int used_buffer_size = current_ * batch_data_size_;
  if (used_buffer_size < cache_file_data_size_)
    return;
  memmove(buffer_, buffer_ + used_buffer_size, total_ - used_buffer_size);

  fill_buffer(total_ - used_buffer_size);

  start_ += current_;
  current_ = 0;
}

void RingBuffer::read_batch_data(int idx, shared_ptr<VariableBuffer> v) {
  int block_size = batch_data_size_;
  if (idx <= prev_index_) {
    // This should not happen, if happen,
    // it means integer overflow, we reset start_
    start_ = idx;
    current_ = 0;
  }
  prev_index_ = idx;
  char *src_buffer = buffer_ + (idx - start_) * block_size;
  v->from_buffer(src_buffer, data_type_, block_size, shape_);
  ++current_;
}

Shape_t RingBuffer::get_shape() const { return shape_; }

dtypes RingBuffer::get_data_type() const { return data_type_; }

DatasetNpyCache::DatasetNpyCache(const ::Dataset &dataset)
    : DatasetImpl(dataset), data_names_(), num_of_data_(0), ring_buffers_() {

  vector<shared_ptr<CacheFile>> cache_files;
  vector<int> idx_list;

  // Read variable names
  load_variable_list(this->cache_dir(), data_names_);

  // Prepare cache file objects
  search_for_cache_files(this->cache_dir(), data_names_, cache_files);

  for (auto c : cache_files) {
    c->preload();
    num_of_data_ += c->get_num_data();
  }

  assert(cache_files.size() > 0);

  if (this->shuffle()) {
    int data_num = cache_files[0]->get_num_data();
    idx_list.resize(data_num);
    std::iota(idx_list.begin(), idx_list.end(), 0);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(std::begin(cache_files), std::end(cache_files),
                 std::default_random_engine(seed));
    std::shuffle(std::begin(idx_list), std::end(idx_list),
                 std::default_random_engine(seed));
  }

  for (auto v : data_names_) {
    ring_buffers_[v] = make_shared<RingBuffer>(cache_files, this->batch_size(),
                                               v, idx_list, this->shuffle());
  }
}

unordered_map<string, shared_ptr<VariableBuffer>>
DatasetNpyCache::get_batch_data(uint32_t iter_num) {
  unordered_map<string, shared_ptr<VariableBuffer>> batch_data;

  for (auto n : data_names_) {
    shared_ptr<VariableBuffer> v = make_shared<VariableBuffer>();
    ring_buffers_[n]->read_batch_data(iter_num, v);
    ring_buffers_[n]->fill_up();
    batch_data[n] = v;
  }

  return batch_data;
}

int DatasetNpyCache::get_num_data() const { return num_of_data_; }

vector<string> DatasetNpyCache::get_data_names() { return data_names_; }

} // namespace nnp
} // namespace utils
} // namespace nbla