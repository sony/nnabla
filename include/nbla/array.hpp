// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

/**
Array interface class and its implementations such as CpuArray, CudaArray
etc.
Array classes are not directly used by users
*/
#ifndef __NBLA_ARRAY_HPP__
#define __NBLA_ARRAY_HPP__

#include <nbla/common.hpp>
#include <nbla/exception.hpp>

namespace nbla {

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
  // HALF,
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
  }
  if (s == 0) {
    NBLA_ERROR(error_code::type, "Unsupported type: %s",
               dtype_to_string(dtype).c_str());
  }
  return s;

#undef GET_DTYPE_SIZE
#undef GET_DTYPE_SIZE_UNSUPPORTED
}

/** An abstract class of Array interface.

This is extended to implement a new array class (see CpuArray, CudaArray etc.).

NOTE for developers: Do not directly access object_ from anywhere except
`allocate` and `deallocate` functions. Use `pointer<T>()` or
`const_pointer<T>()`. Objects are layzily allocated when they are called.
*/
class Array {
protected:
  void *object_; ///< In many case, this is a pointer to device memory.
  Size_t size_;  ///< Size of array.
  dtypes dtype_; ///< Data type such as float32, int32, uint8 etc.
  Context ctx_;  ///< Hold device info to identify
                 ///< what device array is used.
public:
  explicit NBLA_API Array(const Size_t size, dtypes dtype, const Context &ctx);
  virtual NBLA_API ~Array() = 0;

  /** Get object pointer.
   */
  template <typename T = void> T *pointer() {
    if (!object_) {
      allocate();
    }
    return static_cast<T *>(object_);
  }

  /** Get constant object pointer
   */
  template <typename T = void> const T *const_pointer() const {
    if (!object_) {
      const_cast<Array *>(this)->allocate();
    }
    return static_cast<const T *>(object_);
  }

  /** Return dtype. */
  inline dtypes dtype() const { return dtype_; }
  /** Return size of descendant dimensions of specified axis.
  */
  inline Size_t size() const { return size_; }

  /** Return context.
  */
  inline Context context() const { return ctx_; }

  /** Copy from Array. */
  virtual void copy_from(const Array *src_array) = 0;

  /**
    Fill all element with zero.
  */
  virtual void zero() = 0;

  /** Fill all element with given value.
  */
  virtual void fill(float value) = 0;

  /** Filter a Context into a minimal information to describe an Array.
  */
  static Context filter_context(const Context &ctx);

protected:
  /** Memory allocator.
  */
  virtual void allocate() = 0;
  /** Memory deallocator.
  */
  virtual void deallocate() = 0;

  DISABLE_COPY_AND_ASSIGN(Array);
};

/*@}*/
/** \defgroup ArrayImplGrp Array list */
/*@{*/
/*@}*/

// --------------------------------------------------------------------------
// Helper macro for defining copy function from one to another type
#define NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, type, src_type, dst_type)      \
  case dtypes::type:                                                           \
    copy_func<src_type, dst_type>(src_array, this);                            \
    break

#define NBLA_ARRAY_COPY_FROM(copy_func, type)                                  \
  switch (this->dtype()) {                                                     \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, UBYTE, type, unsigned char);       \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, BYTE, type, char);                 \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, USHORT, type, unsigned short);     \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, SHORT, type, short);               \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, UINT, type, unsigned int);         \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, INT, type, int);                   \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, ULONG, type, unsigned long);       \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, LONG, type, long);                 \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, ULONGLONG, type,                   \
                                 unsigned long long);                          \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, LONGLONG, type, long long);        \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, FLOAT, type, float);               \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, DOUBLE, type, double);             \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, BOOL, type, bool);                 \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, LONGDOUBLE, type, long double);    \
  default:                                                                     \
    NBLA_ERROR(error_code::unclassified, "Unknown dtype.");                    \
  }

#define NBLA_CASE_ARRAY_COPY_FROM(copy_func, type, src_type)                   \
  case dtypes::type:                                                           \
    NBLA_ARRAY_COPY_FROM(copy_func, src_type);                                 \
    break;
// --------------------------------------------------------------------------

#define NBLA_DEFINE_FUNC_COPY_FROM(array_class, copy_func)                     \
  void array_class::copy_from(const Array *src_array) {                        \
    if (src_array->size() != this->size_) {                                    \
      NBLA_ERROR(error_code::unclassified, "Size mismatch.");                  \
    }                                                                          \
    switch (src_array->dtype()) {                                              \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, UBYTE, unsigned char);              \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, BYTE, char);                        \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, USHORT, unsigned short);            \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, SHORT, short);                      \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, UINT, unsigned int);                \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, INT, int);                          \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, ULONG, unsigned long);              \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, LONG, long);                        \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, ULONGLONG, unsigned long long);     \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, LONGLONG, long long);               \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, FLOAT, float);                      \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, DOUBLE, double);                    \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, BOOL, bool);                        \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, LONGDOUBLE, long double);           \
    default:                                                                   \
      NBLA_ERROR(error_code::type, "Unknown dtype.");                          \
    }                                                                          \
  }

#define NBLA_CASE_ARRAY_FILL(fill_func, type_enum, type)                       \
  case dtypes::type_enum:                                                      \
    fill_func<type>(this, value);                                              \
    break;
// --------------------------------------------------------------------------

#define NBLA_DEFINE_FUNC_FILL(array_class, fill_func)                          \
  void array_class::fill(float value) {                                        \
    switch (this->dtype()) {                                                   \
      NBLA_CASE_ARRAY_FILL(fill_func, UBYTE, unsigned char);                   \
      NBLA_CASE_ARRAY_FILL(fill_func, BYTE, char);                             \
      NBLA_CASE_ARRAY_FILL(fill_func, USHORT, unsigned short);                 \
      NBLA_CASE_ARRAY_FILL(fill_func, SHORT, short);                           \
      NBLA_CASE_ARRAY_FILL(fill_func, UINT, unsigned int);                     \
      NBLA_CASE_ARRAY_FILL(fill_func, INT, int);                               \
      NBLA_CASE_ARRAY_FILL(fill_func, ULONG, unsigned long);                   \
      NBLA_CASE_ARRAY_FILL(fill_func, LONG, long);                             \
      NBLA_CASE_ARRAY_FILL(fill_func, ULONGLONG, unsigned long long);          \
      NBLA_CASE_ARRAY_FILL(fill_func, LONGLONG, long long);                    \
      NBLA_CASE_ARRAY_FILL(fill_func, FLOAT, float);                           \
      NBLA_CASE_ARRAY_FILL(fill_func, DOUBLE, double);                         \
      NBLA_CASE_ARRAY_FILL(fill_func, BOOL, bool);                             \
      NBLA_CASE_ARRAY_FILL(fill_func, LONGDOUBLE, long double);                \
    default:                                                                   \
      NBLA_ERROR(error_code::type, "Unknown dtype.");                          \
    }                                                                          \
  }
}

// --------------------------------------------------------------------------
// Disable inter-array copy of specific types by template specialization.
#define NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, src_type,      \
                                        dst_type)                              \
  template <>                                                                  \
  void copy_func<src_type, dst_type>(const Array *src, Array *dst) {           \
    NBLA_ERROR(error_code::not_implemented,                                    \
               "Copy from " #src_type " to " #dst_type                         \
               " is disabled in " #array_class ".");                           \
  }

#define NBLA_ARRAY_DISABLE_COPY(array_class, copy_func, type)                  \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type,                \
                                  unsigned char);                              \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type, char);         \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type,                \
                                  unsigned short);                             \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type, short);        \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type, unsigned int); \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type, int);          \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type,                \
                                  unsigned long);                              \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type, long);         \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type,                \
                                  unsigned long long);                         \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type, long long);    \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type, float);        \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type, double);       \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type, long double);  \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, type, bool);         \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, unsigned char,       \
                                  type);                                       \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, char, type);         \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, unsigned short,      \
                                  type);                                       \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, short, type);        \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, unsigned int, type); \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, int, type);          \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, unsigned long,       \
                                  type);                                       \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, long, type);         \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, unsigned long long,  \
                                  type);                                       \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, long long, type);    \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, float, type);        \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, double, type);       \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, long double, type);  \
  NBLA_ARRAY_DISABLE_COPY_FROM_TO(array_class, copy_func, bool, type);

// --------------------------------------------------------------------------
// Disable array filling of a specific type by template specialization.
#define NBLA_ARRAY_DISABLE_FILL(array_class, fill_func, type)                  \
  template <> void fill_func<type>(Array * self, float value) {                \
    NBLA_ERROR(error_code::not_implemented,                                    \
               "Fill function of " #type " is disabled in " #array_class "."); \
  }
#endif
