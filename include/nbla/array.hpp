// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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
#include <nbla/dtypes.hpp>
#include <nbla/event.hpp>
#include <nbla/exception.hpp>
#include <nbla/half.hpp>
#include <nbla/memory/allocator.hpp>

#include <memory>
#include <type_traits>

namespace nbla {

/** \addtogroup NNablaCoreGrp */
/*@{*/

/** An abstract class of Array interface.

This is extended to implement a new array class (see CpuArray, CudaArray etc.).
*/
class Array : public std::enable_shared_from_this<Array> {
protected:
  /// Size of array.
  Size_t size_;

  /// Data type such as float32, int32, uint8 etc.
  dtypes dtype_;

  /// Hold device info to identify what device array is used.
  Context ctx_;

  /// Holding nbla::Memory object.
  AllocatorMemory mem_;

  /// Holding nbla::Event object to wait asynchronous memory copy.
  EventPtr event_;

  /** The constructor must be called only from derived class.

      AllocatorMemory should be created by an instance of a realization of
      nbla::Allocator and passed as a rvalue reference.
   */
  NBLA_API Array(const Size_t size, dtypes dtype, const Context &ctx,
                 AllocatorMemory &&mem);

  static NBLA_API size_t size_as_bytes(Size_t size, dtypes dtype);

  // This method is used to override Array::pointer.
  virtual NBLA_API void *mem_pointer() { return mem_.pointer(); }

  // This method is used to override Array::const_pointer.
  virtual NBLA_API const void *mem_const_pointer() const {
    return mem_.const_pointer();
  }

public:
  typedef shared_ptr<Array> Ptr;

  virtual NBLA_API ~Array() = 0;

  /** Get object pointer.
   */
  template <typename T = void> T *pointer() {
    return reinterpret_cast<T *>(mem_pointer());
  }

  /** Get constant object pointer
   */
  template <typename T = void> const T *const_pointer() const {
    return reinterpret_cast<const T *>(mem_const_pointer());
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
  virtual NBLA_API void zero() = 0;

  /** Fill all element with given value.
  */
  virtual NBLA_API void fill(float value) = 0;

  /** Filter a Context into a minimal information to describe an Array.
  */
  static Context filter_context(const Context &ctx);

  /** Set an event
  */
  virtual NBLA_API void set_event(EventPtr eptr);

  /** Wait for the end of an event

      @param ctx Context where the event is waited for.
      @param async_flags .
  */
  virtual NBLA_API void wait_event(const Context ctx,
                                   const int async_flags = AsyncFlag::NONE);

  /** Check an event exist
  */
  virtual NBLA_API bool have_event();

  /** Get shared_ptr of this object
  */
  virtual NBLA_API Ptr getptr();

protected:
  DISABLE_COPY_AND_ASSIGN(Array);
};

///< Shared pointer of NdArray
typedef Array::Ptr ArrayPtr;

typedef const Array ConstArray;
typedef shared_ptr<ConstArray> ConstArrayPtr;

/*@}*/
/** \defgroup ArrayImplGrp Array list */
/*@{*/
/*@}*/

// --------------------------------------------------------------------------
// Helper macro for defining copy function from one to another type
#define NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, type, src_type, dst_type,      \
                                     name)                                     \
  case dtypes::type:                                                           \
    copy_func##_wrapper<src_type, dst_type>::copy(src_array, this);            \
    break

#define NBLA_ARRAY_COPY_FROM(copy_func, type, name)                            \
  switch (this->dtype()) {                                                     \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, UBYTE, type, unsigned char, name); \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, BYTE, type, char, name);           \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, USHORT, type, unsigned short,      \
                                 name);                                        \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, SHORT, type, short, name);         \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, UINT, type, unsigned int, name);   \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, INT, type, int, name);             \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, ULONG, type, unsigned long, name); \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, LONG, type, long, name);           \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, ULONGLONG, type,                   \
                                 unsigned long long, name);                    \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, LONGLONG, type, long long, name);  \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, FLOAT, type, float, name);         \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, DOUBLE, type, double, name);       \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, BOOL, type, bool, name);           \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, LONGDOUBLE, type, long double,     \
                                 name);                                        \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, HALF, type, nbla::Half, name);     \
  default:                                                                     \
    NBLA_ERROR(error_code::unclassified, "Disabled dtype %s.",                 \
               dtype_to_string(this->dtype()).c_str());                        \
  }

#define NBLA_CASE_ARRAY_COPY_FROM(copy_func, type, src_type, name)             \
  case dtypes::type:                                                           \
    NBLA_ARRAY_COPY_FROM(copy_func, src_type, name);                           \
    break
// --------------------------------------------------------------------------

#define NBLA_DEFINE_FUNC_COPY_FROM(array_class, copy_func, name)               \
  void array_class::copy_from(const Array *src_array) {                        \
    if (src_array->size() != this->size_) {                                    \
      NBLA_ERROR(error_code::unclassified, "Size mismatch.");                  \
    }                                                                          \
    switch (src_array->dtype()) {                                              \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, UBYTE, unsigned char, name);        \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, BYTE, char, name);                  \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, USHORT, unsigned short, name);      \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, SHORT, short, name);                \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, UINT, unsigned int, name);          \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, INT, int, name);                    \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, ULONG, unsigned long, name);        \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, LONG, long, name);                  \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, ULONGLONG, unsigned long long,      \
                                name);                                         \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, LONGLONG, long long, name);         \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, FLOAT, float, name);                \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, DOUBLE, double, name);              \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, BOOL, bool, name);                  \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, LONGDOUBLE, long double, name);     \
      NBLA_CASE_ARRAY_COPY_FROM(copy_func, HALF, nbla::Half, name);            \
    default:                                                                   \
      NBLA_ERROR(error_code::unclassified, "Disabled dtype %s.",               \
                 dtype_to_string(src_array->dtype()).c_str());                 \
    }                                                                          \
  }

#define NBLA_CASE_ARRAY_FILL(fill_func, type_enum, type, name)                 \
  case dtypes::type_enum:                                                      \
    fill_func<type>(this, value);                                              \
    break
// --------------------------------------------------------------------------

#define NBLA_DEFINE_FUNC_FILL(array_class, fill_func, name)                    \
  void array_class::fill(float value) {                                        \
    switch (this->dtype()) {                                                   \
      NBLA_CASE_ARRAY_FILL(fill_func, UBYTE, unsigned char, name);             \
      NBLA_CASE_ARRAY_FILL(fill_func, BYTE, char, name);                       \
      NBLA_CASE_ARRAY_FILL(fill_func, USHORT, unsigned short, name);           \
      NBLA_CASE_ARRAY_FILL(fill_func, SHORT, short, name);                     \
      NBLA_CASE_ARRAY_FILL(fill_func, UINT, unsigned int, name);               \
      NBLA_CASE_ARRAY_FILL(fill_func, INT, int, name);                         \
      NBLA_CASE_ARRAY_FILL(fill_func, ULONG, unsigned long, name);             \
      NBLA_CASE_ARRAY_FILL(fill_func, LONG, long, name);                       \
      NBLA_CASE_ARRAY_FILL(fill_func, ULONGLONG, unsigned long long, name);    \
      NBLA_CASE_ARRAY_FILL(fill_func, LONGLONG, long long, name);              \
      NBLA_CASE_ARRAY_FILL(fill_func, FLOAT, float, name);                     \
      NBLA_CASE_ARRAY_FILL(fill_func, DOUBLE, double, name);                   \
      NBLA_CASE_ARRAY_FILL(fill_func, BOOL, bool, name);                       \
      NBLA_CASE_ARRAY_FILL(fill_func, LONGDOUBLE, long double, name);          \
      NBLA_CASE_ARRAY_FILL(fill_func, HALF, nbla::Half, name);                 \
    default:                                                                   \
      NBLA_ERROR(error_code::unclassified, "Disabled dtype %s.",               \
                 dtype_to_string(this->dtype()).c_str());                      \
    }                                                                          \
  }
}
#define NBLA_DEFINE_COPY_WRAPPER(copy_func)                                    \
  template <typename Ta, typename Tb, typename Enabled = void>                 \
  struct copy_func##_wrapper {                                                 \
    static void copy(const Array *src, Array *dst) {                           \
      copy_func<Ta, Tb>(src, dst);                                             \
    }                                                                          \
  };                                                                           \
  template <typename T> struct copy_func##_is_disabled {                       \
    static constexpr bool value = false;                                       \
  }

#define NBLA_DISABLE_TYPE(copy_func, fill_func, TYPE)                          \
  template <typename Ta, typename Tb>                                          \
  struct copy_func##_wrapper<                                                  \
      Ta, Tb, typename std::enable_if<std::is_same<Ta, TYPE>::value>::type> {  \
    static void copy(const Array *src, Array *dst) {                           \
      NBLA_ERROR(error_code::not_implemented,                                  \
                 "`" #TYPE "` is disabled in `" #copy_func "`.");              \
    }                                                                          \
  };                                                                           \
  template <> struct copy_func##_is_disabled<TYPE> {                           \
    static constexpr bool value = true;                                        \
  };                                                                           \
  template <typename Ta, typename Tb>                                          \
  struct copy_func##_wrapper<                                                  \
      Ta, Tb, typename std::enable_if<!copy_func##_is_disabled<Ta>::value &&   \
                                      std::is_same<Tb, TYPE>::value>::type> {  \
    static void copy(const Array *src, Array *dst) {                           \
      NBLA_ERROR(error_code::not_implemented,                                  \
                 "`" #TYPE "` is disabled in `" #copy_func "`.");              \
    }                                                                          \
  };                                                                           \
  template <> void fill_func<TYPE>(Array * self, float value) {                \
    NBLA_ERROR(error_code::not_implemented,                                    \
               "`" #TYPE "` is disabled in `" #fill_func "`.");                \
  }

#endif
