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

#ifndef NBLA_HALF_HPP_
#define NBLA_HALF_HPP_

#include <nbla/defs.hpp>
#include <nbla/dtypes.hpp>
#include <nbla/exception.hpp>

#include <nbla/float_bits.hpp>

#include <cmath>
#include <limits>

namespace nbla {
inline uint16_t float2halfbits(float fvalue) {
  union {
    uint32_t bits;
    float value_;
  };
  value_ = fvalue;
  return float_bits<float>::downconvert_to<Half>(bits);
}

inline float halfbits2float(uint16_t hbits) {
  union {
    uint32_t fbits;
    float fvalue;
  };
  fbits = float_bits<Half>::upconvert_to<float>(hbits);
  return fvalue;
}

/** \addtogroup NNablaCoreGrp */
/*@{*/

/** Implementation of Half precision float emulating a builtin scalar type.
 */
struct NBLA_ALIGN(2) Half {
  uint16_t bits;

  // Constructors
  Half();
#define CST(TYPE) NBLA_API Half(const TYPE &rhs)
  CST(unsigned char);
  CST(char);
  CST(unsigned short);
  CST(short);
  CST(unsigned int);
  CST(int);
  CST(unsigned long);
  CST(long);
  CST(unsigned long long);
  CST(long long);
  CST(float);
  CST(double);
  CST(bool);
  CST(long double);
  CST(Half);
#undef CST

  // Assignment Operators
  NBLA_API Half &operator+=(const Half &rhs);
  NBLA_API Half &operator-=(const Half &rhs);
  NBLA_API Half &operator*=(const Half &rhs);
  NBLA_API Half &operator/=(const Half &rhs);
  NBLA_API Half &operator=(const Half &rhs);
#define CST(TYPE) NBLA_API operator TYPE() const
  CST(unsigned char);
  CST(char);
  CST(unsigned short);
  CST(short);
  CST(unsigned int);
  CST(int);
  CST(unsigned long);
  CST(long);
  CST(unsigned long long);
  CST(long long);
  CST(float);
  CST(double);
  CST(bool);
  CST(long double);
#undef CST

  // Arithmetic operators
  NBLA_API Half operator+() const;
  NBLA_API Half operator-() const;
#define AOP(OP, TYPE) NBLA_API Half operator OP(TYPE o) const
#define AOPR(OP, TYPE) NBLA_API TYPE operator OP(TYPE o) const
#define AOP_TYPE(OP)                                                           \
  AOP(OP, unsigned char);                                                      \
  AOP(OP, char);                                                               \
  AOP(OP, unsigned short);                                                     \
  AOP(OP, short);                                                              \
  AOP(OP, unsigned int);                                                       \
  AOP(OP, int);                                                                \
  AOP(OP, unsigned long);                                                      \
  AOP(OP, long);                                                               \
  AOP(OP, unsigned long long);                                                 \
  AOP(OP, long long);                                                          \
  AOPR(OP, float);                                                             \
  AOPR(OP, double);                                                            \
  AOP(OP, bool);                                                               \
  AOPR(OP, long double);                                                       \
  AOP(OP, Half)
  AOP_TYPE(+);
  AOP_TYPE(-);
  AOP_TYPE(*);
  AOP_TYPE(/);
#undef AOP_TYPE
#undef AOP
#undef AOPR

  // Inc/dec operators
  // Half& operator ++ ();
  // Half& operator -- ();
  // Half operator ++ (int dummy);
  // Half operator -- (int dummy);
};

// Inverse arithmetic operators
#define AOP(OP, TYPE)                                                          \
  NBLA_API Half operator OP(const TYPE &lhs, const Half &rhs)
#define AOPR(OP, TYPE)                                                         \
  NBLA_API TYPE operator OP(const TYPE &lhs, const Half &rhs)
#define AOP_TYPE(OP)                                                           \
  AOP(OP, unsigned char);                                                      \
  AOP(OP, char);                                                               \
  AOP(OP, unsigned short);                                                     \
  AOP(OP, short);                                                              \
  AOP(OP, unsigned int);                                                       \
  AOP(OP, int);                                                                \
  AOP(OP, unsigned long);                                                      \
  AOP(OP, long);                                                               \
  AOP(OP, unsigned long long);                                                 \
  AOP(OP, long long);                                                          \
  AOPR(OP, float);                                                             \
  AOPR(OP, double);                                                            \
  AOP(OP, bool);                                                               \
  AOPR(OP, long double)
AOP_TYPE(+);
AOP_TYPE(-);
AOP_TYPE(*);
AOP_TYPE(/);
#undef AOP_TYPE
#undef AOP
#undef AOPR

// Relational operators
#define ROP_TYPE(OP, TYPE)                                                     \
  NBLA_API bool operator OP(const Half &lhs, const TYPE &rhs)
#define IROP_TYPE(OP, TYPE)                                                    \
  NBLA_API bool operator OP(const TYPE &lhs, const Half &rhs)
#define ROP(TYPE)                                                              \
  ROP_TYPE(<, TYPE);                                                           \
  ROP_TYPE(>, TYPE);                                                           \
  ROP_TYPE(<=, TYPE);                                                          \
  ROP_TYPE(>=, TYPE);                                                          \
  ROP_TYPE(==, TYPE);                                                          \
  ROP_TYPE(!=, TYPE);                                                          \
  IROP_TYPE(<, TYPE);                                                          \
  IROP_TYPE(>, TYPE);                                                          \
  IROP_TYPE(<=, TYPE);                                                         \
  IROP_TYPE(>=, TYPE)
ROP(unsigned char);
ROP(char);
ROP(unsigned short);
ROP(short);
ROP(unsigned int);
ROP(int);
ROP(unsigned long);
ROP(long);
ROP(unsigned long long);
ROP(long long);
ROP(float);
ROP(double);
ROP(bool);
ROP(long double);
ROP_TYPE(<, Half);
ROP_TYPE(>, Half);
ROP_TYPE(<=, Half);
ROP_TYPE(>=, Half);
ROP_TYPE(==, Half);
ROP_TYPE(!=, Half);

#undef ROP_TYPE
#undef IROP_TYPE
#undef ROP

template <typename T> struct force_float { typedef T type; };
template <> struct force_float<Half> { typedef float type; };

/*@}*/
}
// cmath functions

namespace std {
using namespace nbla;
#define MATHF(FUNC) NBLA_API Half FUNC(const Half &h)
MATHF(exp);
MATHF(log);
MATHF(log2);
MATHF(log1p);
MATHF(sqrt);
MATHF(sin);
MATHF(cos);
MATHF(tan);
MATHF(sinh);
MATHF(cosh);
MATHF(tanh);
MATHF(asin);
MATHF(acos);
MATHF(atan);
MATHF(asinh);
MATHF(acosh);
MATHF(atanh);
MATHF(fabs);
MATHF(abs);
MATHF(floor);
MATHF(ceil);
MATHF(round);
MATHF(isnan);
MATHF(isinf);
NBLA_API Half pow(const Half &a, const Half &b);
NBLA_API Half pow(const Half &a, const int &b);
NBLA_API Half max(const Half &a, const int &b);
NBLA_API Half atan2(const Half &a, const Half &b);
NBLA_API Half ldexp(const Half &a, const int &b);
#undef MATHF

template <> class numeric_limits<Half> {
public:
  inline static Half min() { return 6.10352e-5; }
  inline static Half max() { return 3.2768e+4; }
  static constexpr bool is_integer = false;
  static constexpr bool is_signed = true;
};
}
#endif
