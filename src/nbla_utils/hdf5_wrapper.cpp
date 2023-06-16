// Copyright 2023 Sony Group Corporation.
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
#include <assert.h>
#include <fstream>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <iostream>
#include <random>

#include "hdf5_wrapper.hpp"
#include "parameters_impl.hpp"
#include <nbla/computation_graph/function.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/logger.hpp>

using namespace std;
namespace nbla {
namespace utils {

const nbla::Context kCpuCtx{{"cpu:float"}, "CpuCachedArray", "0"};
constexpr int MAX_NAME = 1024;

#define CHECK_FUNC_RET(f)                                                      \
  if (f < 0) {                                                                 \
    return false;                                                              \
  } else {                                                                     \
    unknown = false;                                                           \
  }

#define COMPARE_AND_READ(ttid, t)                                              \
  if (H5Tequal(tid, ttid)) {                                                   \
    t *buffer =                                                                \
        ndarray->cast(nbla::get_dtype<t>(), kCpuCtx, true)->pointer<t>();      \
    CHECK_FUNC_RET(H5Dread(did, ttid, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer)); \
  }

bool read_data(hid_t did, NdArrayPtr ndarray) {
  hid_t sp = H5Dget_space(did);
  int rank = H5Sget_simple_extent_ndims(sp);
  std::vector<hsize_t> dims(rank);
  if (!H5Sget_simple_extent_dims(sp, dims.data(), NULL))
    return false;

  // NdArray ndarray;;
  Shape_t shape_x(rank);
  for (int i = 0; i < rank; i++)
    shape_x[i] = dims[i];
  ndarray->reshape(shape_x, true);

  hid_t tid = H5Dget_type(did);

  hid_t H5T_CUSTOM_F16BE = H5Tcopy(H5T_IEEE_F32BE);
  H5Tset_fields(H5T_CUSTOM_F16BE, 15, 10, 5, 0, 10);
  H5Tset_size(H5T_CUSTOM_F16BE, 2);
  H5Tset_ebias(H5T_CUSTOM_F16BE, 15);
  H5Tlock(H5T_CUSTOM_F16BE);

  hid_t H5T_CUSTOM_F16LE = H5Tcopy(H5T_CUSTOM_F16BE);
  H5Tset_order(H5T_CUSTOM_F16LE, H5T_ORDER_LE);
  H5Tlock(H5T_CUSTOM_F16LE);

  bool unknown = true;

  COMPARE_AND_READ(H5T_CUSTOM_F16BE, Half);
  COMPARE_AND_READ(H5T_CUSTOM_F16LE, Half);
  COMPARE_AND_READ(H5T_IEEE_F32BE, float);
  COMPARE_AND_READ(H5T_IEEE_F32LE, float);
  COMPARE_AND_READ(H5T_NATIVE_CHAR, char);
  COMPARE_AND_READ(H5T_NATIVE_UCHAR, uint8_t);
  COMPARE_AND_READ(H5T_NATIVE_SHORT, short);
  COMPARE_AND_READ(H5T_NATIVE_USHORT, unsigned short);
  COMPARE_AND_READ(H5T_NATIVE_INT, int);
  COMPARE_AND_READ(H5T_NATIVE_UINT, unsigned int);
  COMPARE_AND_READ(H5T_NATIVE_LONG, long);
  COMPARE_AND_READ(H5T_NATIVE_ULONG, unsigned long);
  COMPARE_AND_READ(H5T_NATIVE_LLONG, long long);
  COMPARE_AND_READ(H5T_NATIVE_ULLONG, unsigned long long);
  COMPARE_AND_READ(H5T_NATIVE_FLOAT, float);
  COMPARE_AND_READ(H5T_NATIVE_DOUBLE, double);
  COMPARE_AND_READ(H5T_NATIVE_LDOUBLE, long double);
  COMPARE_AND_READ(H5T_NATIVE_HBOOL, bool);

  if (unknown) {
    NBLA_ERROR(error_code::value, "Unknown data type occurs.");
  }

  return true;
}

bool read_dataset(hid_t gid, const int id, string &data_name,
                  NdArrayPtr ndarray) {
  char name[MAX_NAME];

  if (H5Gget_objname_by_idx(gid, (hsize_t)id, name, (size_t)MAX_NAME) < 0)
    return false;
  data_name = string(name);

  hid_t did = H5Dopen(gid, name, H5P_DEFAULT);
  bool ret = read_data(did, ndarray);
  H5Dclose(did);
  return ret;
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
  for (size_t i = 0; i < num; i++) {
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

  for (size_t i = 0; i < dataset_ids.size(); i++) {
    read_dataset(gid, dataset_ids[i], data_names[i], ndarrays[i]);
  }

  return true;
}

bool parse_hdf5_dataset(string name, hid_t did, ParameterVector &pv) {
  string variable_name = name.substr(1, name.length());
  auto ndarray = make_shared<NdArray>();
  bool ret = read_data(did, ndarray);
  if (!ret) {
    return ret;
  }
  int need_grad = false; // default need_grad
  if (H5Aexists(did, "need_grad")) {
    hid_t att = H5Aopen(did, "need_grad", H5P_DEFAULT);
    H5Aread(att, H5T_NATIVE_HBOOL, &need_grad);
    H5Aclose(att);
  }
  VariablePtr v = make_shared<Variable>(ndarray);
  CgVariablePtr cg_v = make_shared<CgVariable>(v, need_grad);
  pv.push_back({variable_name, cg_v});
  return true;
}

bool parse_hdf5_group(hid_t gid, ParameterVector &pv) {
  ssize_t len;
  hsize_t num = 0;
  herr_t err = H5Gget_num_objs(gid, &num);
  if (err >= 0) {
    char group_name[MAX_NAME];
    len = H5Iget_name(gid, group_name, MAX_NAME);
    if (len >= 0) {
      for (unsigned int i = 0; i < num; i++) {
        char name[MAX_NAME];
        len = H5Gget_objname_by_idx(gid, (hsize_t)i, name, (size_t)MAX_NAME);
        if (len < 0) {
          return false;
        }

        int type = H5Gget_objtype_by_idx(gid, i);
        switch (type) {
        case H5G_GROUP: {
          hid_t grpid = H5Gopen(gid, name, H5P_DEFAULT);
          parse_hdf5_group(grpid, pv);
          H5Gclose(grpid);
          break;
        }
        case H5G_DATASET: {
          hid_t did = H5Dopen(gid, name, H5P_DEFAULT);
          string dataset_name(group_name);
          if (dataset_name != "/")
            dataset_name += "/";
          dataset_name += string(name);
          parse_hdf5_dataset(dataset_name, did, pv);
          H5Dclose(did);
          break;
        }
        case H5G_TYPE:
          NBLA_LOG_INFO("H5G_TYPE");
          break;
        case H5G_LINK:
          NBLA_LOG_INFO("H5G_LINK");
          break;
        default:
          // TODO: Unsupported member.
          NBLA_LOG_INFO("default");
          break;
        }
      }
      return true;
    }
  }
  return false;
}

void create_h5_group(hid_t file_id, const string &filename) {
  hid_t group_id;
  int ep = filename.find_last_of("/");
  if (ep == -1) {
    return;
  } else {
    H5G_info_t group_info;
    string base_name = filename.substr(0, ep);
    herr_t err = H5Gget_info_by_name(file_id, base_name.c_str(), &group_info,
                                     H5P_DEFAULT);
    if (err >= 0) {
      return;
    } else {
      create_h5_group(file_id, base_name);
      group_id = H5Gcreate2(file_id, base_name.c_str(), H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);
      if (group_id < 0) {
        NBLA_ERROR(error_code::value, "Cannot create dataset %s in .h5 file.",
                   base_name.c_str());
      } else {
        H5Gclose(group_id);
      }
    }
  }
}

#define WRITE_BY_TYPE(dt, t, ht)                                               \
  case dt: {                                                                   \
    t *data = variable->template cast_data_and_get_pointer<t>(kCpuCtx);        \
    status = H5Dwrite(dataset_id, ht, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);    \
  } break;

void create_h5_dataset(const ParameterVector &pv, hid_t file_id) {
  hid_t dataset_id, dataspace_id;
  hid_t attr_space_id, attr_id;
  herr_t status;
  int index = 0;

  hid_t H5T_CUSTOM_F16BE = H5Tcopy(H5T_IEEE_F32BE);
  H5Tset_fields(H5T_CUSTOM_F16BE, 15, 10, 5, 0, 10);
  H5Tset_size(H5T_CUSTOM_F16BE, 2);
  H5Tset_ebias(H5T_CUSTOM_F16BE, 15);
  H5Tlock(H5T_CUSTOM_F16BE);

  hid_t H5T_CUSTOM_F16LE = H5Tcopy(H5T_CUSTOM_F16BE);
  H5Tset_order(H5T_CUSTOM_F16LE, H5T_ORDER_LE);
  H5Tlock(H5T_CUSTOM_F16LE);

  for (auto it = pv.begin(); it != pv.end(); it++, index++) {
    string name = it->first;
    CgVariablePtr cg_v = it->second;
    VariablePtr variable = cg_v->variable();
    Shape_t shape = variable->shape();
    dataspace_id =
        H5Screate_simple(shape.size(), (hsize_t *)shape.data(), NULL);
    create_h5_group(file_id, name);
    dataset_id =
        H5Dcreate2(file_id, name.c_str(), H5T_NATIVE_FLOAT, dataspace_id,
                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    switch (cg_v->variable()->data()->array()->dtype()) {

      WRITE_BY_TYPE(dtypes::HALF, Half, H5T_CUSTOM_F16LE);
      WRITE_BY_TYPE(dtypes::FLOAT, float, H5T_IEEE_F32LE);
      WRITE_BY_TYPE(dtypes::SHORT, short, H5T_NATIVE_SHORT);
      WRITE_BY_TYPE(dtypes::USHORT, unsigned short, H5T_NATIVE_USHORT);
      WRITE_BY_TYPE(dtypes::BYTE, char, H5T_NATIVE_CHAR);
      WRITE_BY_TYPE(dtypes::UBYTE, unsigned char, H5T_NATIVE_UCHAR);
      WRITE_BY_TYPE(dtypes::INT, int, H5T_NATIVE_INT);
      WRITE_BY_TYPE(dtypes::UINT, unsigned int, H5T_NATIVE_UINT);
      WRITE_BY_TYPE(dtypes::LONG, long, H5T_NATIVE_LONG);
      WRITE_BY_TYPE(dtypes::ULONG, unsigned long, H5T_NATIVE_ULONG);
      WRITE_BY_TYPE(dtypes::LONGLONG, long long, H5T_NATIVE_LLONG);
      WRITE_BY_TYPE(dtypes::ULONGLONG, unsigned long long, H5T_NATIVE_ULLONG);
      WRITE_BY_TYPE(dtypes::DOUBLE, double, H5T_IEEE_F64LE);
      WRITE_BY_TYPE(dtypes::LONGDOUBLE, long double, H5T_NATIVE_LDOUBLE);
      WRITE_BY_TYPE(dtypes::BOOL, bool, H5T_NATIVE_HBOOL);
    }

    if (status >= 0) {
      attr_space_id = H5Screate(H5S_SCALAR);
      int need_grad = cg_v->need_grad();
      attr_id = H5Acreate2(dataset_id, "need_grad", H5T_NATIVE_INT,
                           attr_space_id, H5P_DEFAULT, H5P_DEFAULT);
      H5Awrite(attr_id, H5T_NATIVE_INT, &need_grad);
      H5Aclose(attr_id);
      H5Sclose(attr_space_id);
      attr_space_id = H5Screate(H5S_SCALAR);
      attr_id = H5Acreate2(dataset_id, "index", H5T_NATIVE_INT, attr_space_id,
                           H5P_DEFAULT, H5P_DEFAULT);
      H5Awrite(attr_id, H5T_NATIVE_INT, &index);
      H5Aclose(attr_id);
      H5Sclose(attr_space_id);
    } else {
      NBLA_ERROR(error_code::not_implemented,
                 "Cannot write h5 file's dataset.");
    }
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
  }
}

bool read_file_to_str(string h5file, std::vector<char> &h5data) {
  std::ifstream file(h5file.c_str(), std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Error: could not open file " << h5file << std::endl;
    return false;
  }

  size_t filesize = static_cast<size_t>(file.tellg());
  if (filesize <= 0) {
    file.close();
    std::cerr << "Error: file size 0" << std::endl;
    return false;
  }

  h5data.resize(filesize);
  file.seekg(0, file.beg);
  file.read(h5data.data(), filesize);
  file.close();

  return true;
}

// ----------------------------------------------------------------------
// load ndarrays from .h5 file
// ----------------------------------------------------------------------
bool load_from_h5_file(string h5file, vector<string> &data_names,
                       vector<NdArrayPtr> &ndarrays) {
  std::vector<char> h5data;
  if (!read_file_to_str(h5file, h5data)) {
    return false;
  }
  return load_from_h5_buffer((char *)h5data.data(), h5data.size(), data_names,
                             ndarrays);
}

// ----------------------------------------------------------------------
// load ndarrays from .h5 file buffer
// ----------------------------------------------------------------------
bool load_from_h5_buffer(char *buffer, size_t size, vector<string> &data_names,
                         vector<NdArrayPtr> &ndarrays) {
  hid_t id = H5LTopen_file_image(buffer, size, H5LT_FILE_IMAGE_DONT_RELEASE);
  if (0 <= id) {
    hid_t root = H5Gopen(id, "/", H5P_DEFAULT);
    if (0 <= root)
      parse_hdf5_group(root, data_names, ndarrays);
    H5Fclose(id);
  } else {
    NBLA_ERROR(error_code::value, "Cannot open image file.");
    return false;
  }
  return true;
}

// ----------------------------------------------------------------------
// load ParameterVector from .h5 file
// ----------------------------------------------------------------------
bool load_parameters_h5(ParameterVector &pv, string h5file) {
  vector<char> h5data;
  if (!read_file_to_str(h5file, h5data)) {
    return false;
  }
  return load_parameters_h5(pv, h5data.data(), h5data.size());
}

// ----------------------------------------------------------------------
// load ParameterVector from .h5 file buffer
// ----------------------------------------------------------------------
bool load_parameters_h5(ParameterVector &pv, char *buffer, size_t size) {
  hid_t id = H5LTopen_file_image(buffer, size, H5LT_FILE_IMAGE_DONT_RELEASE);
  if (0 <= id) {
    hid_t root = H5Gopen(id, "/", H5P_DEFAULT);
    if (0 <= root) {
      parse_hdf5_group(root, pv);
      H5Gclose(root);
    } else {
      NBLA_ERROR(error_code::value, "Cannot found group in buffer.");
    }
    H5Fclose(id);
  } else {
    NBLA_ERROR(error_code::value, "Cannot open image file.");
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------
// generate a random string for temporary file name.
// Since tmpnam is not safe, and mkstemp is not what I want.
// ----------------------------------------------------------------------
nbla::string random_string(void) {
  int random_string_len = 10;

  static auto &stuff = "0123456789_"
                       "abcdefghijklmnopqrstuvwxy"
                       "ABCDEFGHIJKLMNOPQRSTUVWXY"; // No 'Z'
  thread_local static std::uniform_int_distribution<std::string::size_type>
      pick(0, sizeof(stuff) - 2);
  thread_local static std::mt19937 rg{std::random_device{}()};
  nbla::string random_str;
  random_str.resize(random_string_len);
  for (int i = 0; i < random_string_len; ++i)
    random_str[i] = stuff[pick(rg)];
  return random_str;
}

// ----------------------------------------------------------------------
// save parameters to h5 file buffer
// ----------------------------------------------------------------------
bool save_parameters_h5(const ParameterVector &pv, char *buffer,
                        unsigned int &size) {
  hid_t file_id;
  nbla::string filename;
#ifndef WIN32
  const char *tmp = getenv("TMPDIR");
#else
  const char *tmp = getenv("TEMP");
#endif

  H5Eset_auto1(NULL, NULL);
  if (tmp == 0)
    filename = "/tmp";
  else
    filename = tmp;
  filename += "/";
  filename += random_string();
  file_id =
      H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  create_h5_dataset(pv, file_id);
  H5Fclose(file_id);
  std::ifstream ifs(filename, ios::binary);
  const auto begin = ifs.tellg();
  ifs.seekg(0, ios::end);
  const auto end = ifs.tellg();
  const auto file_size = end - begin;
  if (buffer == NULL) {
    size = file_size;
    remove(filename.c_str());
    return true;
  } else {
    if (size < file_size) {
      NBLA_ERROR(error_code::memory,
                 "Required memory size %d is not satisfied by %d!", file_size,
                 size);
    }
    ifs.seekg(0, ios::beg);
    ifs.read(buffer, file_size);
    size = file_size;
  }
  remove(filename.c_str());
  return true;
}

// ----------------------------------------------------------------------
// save parameters to a .h5 file specified by filename
// ----------------------------------------------------------------------
bool save_parameters_h5(const ParameterVector &pv, string filename) {
  hid_t file_id;

  H5Eset_auto1(NULL, NULL);
  file_id =
      H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  create_h5_dataset(pv, file_id);
  H5Fclose(file_id);
  return true;
}
} // namespace utils
} // namespace nbla
