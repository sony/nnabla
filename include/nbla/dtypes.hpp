// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#ifndef NBLA_DTYPES_HPP_
#define NBLA_DTYPES_HPP_

#include <nbla/exception.hpp>

namespace nbla {

// Forward declaration
struct Half;

/** \addtogroup NNablaCoreGrp */
/*@{*/

/** ENUM for dtypes

Compatible with numpy DTYPES. It allows us to use np.dtype in Python interface.
*/
enum class dtypes {
  BOOL = 0,
  BYTE,
  UBYTE,
  SHORT,
  USHORT,
  INT,
  UINT,
  LONG,
  ULONG,
  LONGLONG,
  ULONGLONG,
  FLOAT,
  DOUBLE,
  LONGDOUBLE,
  // // Following items are for compatibility with Numpy
  // CFLOAT,
  // CDOUBLE,
  // CLONGDOUBLE,
  // OBJECT = 17,
  // STRING,
  // UNICODE,
  // VOID,
  // Appended in numpy 1.6
  // DATETIME,
  // TIMEDELTA,
  HALF = 23,
  // NTYPES,
  // NOTYPE,
  // CHAR,
  // USERDEF = 256,
  // NTYPES_ABI_COMPATIBLE = 21
};

/** A function to get dtype enum by dtype of C++.

EX) dtypes dtype = get_dtype<T>::type;
*/
template <typename T> dtypes get_dtype() {
  NBLA_ERROR(error_code::type, "Unsupported dtype.");
}
#define GET_DTYPE_TEMPLATE_SPECIAL(type, Dtype)                                \
  template <> inline dtypes get_dtype<type>() { return dtypes::Dtype; }
GET_DTYPE_TEMPLATE_SPECIAL(unsigned char, UBYTE);
GET_DTYPE_TEMPLATE_SPECIAL(char, BYTE);
GET_DTYPE_TEMPLATE_SPECIAL(unsigned short, USHORT);
GET_DTYPE_TEMPLATE_SPECIAL(short, SHORT);
GET_DTYPE_TEMPLATE_SPECIAL(unsigned int, UINT);
GET_DTYPE_TEMPLATE_SPECIAL(int, INT);
GET_DTYPE_TEMPLATE_SPECIAL(unsigned long, ULONG);
GET_DTYPE_TEMPLATE_SPECIAL(long, LONG);
GET_DTYPE_TEMPLATE_SPECIAL(unsigned long long, ULONGLONG);
GET_DTYPE_TEMPLATE_SPECIAL(long long, LONGLONG);
GET_DTYPE_TEMPLATE_SPECIAL(float, FLOAT);
GET_DTYPE_TEMPLATE_SPECIAL(double, DOUBLE);
GET_DTYPE_TEMPLATE_SPECIAL(bool, BOOL);
GET_DTYPE_TEMPLATE_SPECIAL(long double, LONGDOUBLE);
GET_DTYPE_TEMPLATE_SPECIAL(nbla::Half, HALF);
#undef GET_DTYPE_TEMPLATE_SPECIAL

/// Convert dtypes to string
inline string dtype_to_string(dtypes dtype) {
#define GET_DTYPE_STRING(TYPE)                                                 \
  case dtypes::TYPE:                                                           \
    s = #TYPE;                                                                 \
    break;

  string s;
  switch (dtype) {
    GET_DTYPE_STRING(UBYTE);
    GET_DTYPE_STRING(BYTE);
    GET_DTYPE_STRING(USHORT);
    GET_DTYPE_STRING(SHORT);
    GET_DTYPE_STRING(UINT);
    GET_DTYPE_STRING(INT);
    GET_DTYPE_STRING(ULONG);
    GET_DTYPE_STRING(LONG);
    GET_DTYPE_STRING(ULONGLONG);
    GET_DTYPE_STRING(LONGLONG);
    GET_DTYPE_STRING(FLOAT);
    GET_DTYPE_STRING(DOUBLE);
    GET_DTYPE_STRING(BOOL);
    GET_DTYPE_STRING(LONGDOUBLE);
    GET_DTYPE_STRING(HALF);
  }
  if (s.empty()) {
    NBLA_ERROR(error_code::type, "Unknown dtype %d", int(dtype));
  }
  return s;
#undef GET_DTYPE_STRING
}
/** Figure out the size of given dtype.
 */
inline size_t sizeof_dtype(dtypes dtype) {
//-- macro
#define GET_DTYPE_SIZE(type, TYPE)                                             \
  case dtypes::TYPE:                                                           \
    s = sizeof(type);                                                          \
    break;
  size_t s = 0;
  //--
  switch (dtype) {
    GET_DTYPE_SIZE(unsigned char, UBYTE);
    GET_DTYPE_SIZE(char, BYTE);
    GET_DTYPE_SIZE(unsigned short, USHORT);
    GET_DTYPE_SIZE(short, SHORT);
    GET_DTYPE_SIZE(unsigned int, UINT);
    GET_DTYPE_SIZE(int, INT);
    GET_DTYPE_SIZE(unsigned long, ULONG);
    GET_DTYPE_SIZE(long, LONG);
    GET_DTYPE_SIZE(unsigned long long, ULONGLONG);
    GET_DTYPE_SIZE(long long, LONGLONG);
    GET_DTYPE_SIZE(float, FLOAT);
    GET_DTYPE_SIZE(double, DOUBLE);
    GET_DTYPE_SIZE(bool, BOOL);
    GET_DTYPE_SIZE(long double, LONGDOUBLE);
    GET_DTYPE_SIZE(uint16_t, HALF);
  }
  if (s == 0) {
    NBLA_ERROR(error_code::type, "Unsupported type: %s",
               dtype_to_string(dtype).c_str());
  }
  return s;

#undef GET_DTYPE_SIZE
#undef GET_DTYPE_SIZE_UNSUPPORTED
}

/*@}*/
}
#endif
