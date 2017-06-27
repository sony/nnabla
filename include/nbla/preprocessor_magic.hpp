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

#ifndef __NBLA_PREPROCESSOR_MAGIC_HPP__
#define __NBLA_PREPROCESSOR_MAGIC_HPP__

// Helper start-----------------------------------------------------------------
// Workaround for incomplete empty variadic macro support in MSVC
#define NBLA_VA_ARGS(...) , ##__VA_ARGS__
// Workaround for variadic macro in MSVC. __VA_ARGS__ is interpreted as a single
// argument in nested macro call
#define NBLA_EXPAND(...) __VA_ARGS__
// Helper macros for NBLA_ARGDEFS
#define NBLA_ARGDEFS_0()
#define NBLA_ARGDEFS_1(A1) A1 a1
#define NBLA_ARGDEFS_2(A1, A2) NBLA_ARGDEFS_1(A1), A2 a2
#define NBLA_ARGDEFS_3(A1, A2, A3) NBLA_ARGDEFS_2(A1, A2), A3 a3
#define NBLA_ARGDEFS_4(A1, A2, A3, A4) NBLA_ARGDEFS_3(A1, A2, A3), A4 a4
#define NBLA_ARGDEFS_5(A1, A2, A3, A4, A5) NBLA_ARGDEFS_4(A1, A2, A3, A4), A5 a5
#define NBLA_ARGDEFS_6(A1, A2, A3, A4, A5, A6)                                 \
  NBLA_ARGDEFS_5(A1, A2, A3, A4, A5), A6 a6
#define NBLA_ARGDEFS_7(A1, A2, A3, A4, A5, A6, A7)                             \
  NBLA_ARGDEFS_6(A1, A2, A3, A4, A5, A6), A7 a7
#define NBLA_ARGDEFS_8(A1, A2, A3, A4, A5, A6, A7, A8)                         \
  NBLA_ARGDEFS_7(A1, A2, A3, A4, A5, A6, A7), A8 a8
#define NBLA_ARGDEFS_9(A1, A2, A3, A4, A5, A6, A7, A8, A9)                     \
  NBLA_ARGDEFS_8(A1, A2, A3, A4, A5, A6, A7, A8), A9 a9
#define NBLA_ARGDEFS_10(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)               \
  NBLA_ARGDEFS_9(A1, A2, A3, A4, A5, A6, A7, A8, A9), A10 a10
#define NBLA_ARGDEFS_11(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)          \
  NBLA_ARGDEFS_10(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10), A11 a11
#define NBLA_ARGDEFS_12(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)     \
  NBLA_ARGDEFS_11(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11), A12 a12
#define NBLA_ARGDEFS_13(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13)                                                   \
  NBLA_ARGDEFS_12(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12), A13 a13
#define NBLA_ARGDEFS_14(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14)                                              \
  NBLA_ARGDEFS_13(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13)      \
  , A14 a14
#define NBLA_ARGDEFS_15(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15)                                         \
  NBLA_ARGDEFS_14(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14) \
  , A15 a15
#define NBLA_ARGDEFS_16(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16)                                    \
  NBLA_ARGDEFS_15(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15)                                                         \
  , A16 a16
#define NBLA_ARGDEFS_17(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17)                               \
  NBLA_ARGDEFS_16(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16)                                                    \
  , A17 a17

// Helper macros for NBLA_ARGS
#define NBLA_ARGS_0()
#define NBLA_ARGS_1(A1) a1
#define NBLA_ARGS_2(A1, A2) a1, a2
#define NBLA_ARGS_3(A1, A2, A3) a1, a2, a3
#define NBLA_ARGS_4(A1, A2, A3, A4) a1, a2, a3, a4
#define NBLA_ARGS_5(A1, A2, A3, A4, A5) a1, a2, a3, a4, a5
#define NBLA_ARGS_6(A1, A2, A3, A4, A5, A6) a1, a2, a3, a4, a5, a6
#define NBLA_ARGS_7(A1, A2, A3, A4, A5, A6, A7) a1, a2, a3, a4, a5, a6, a7
#define NBLA_ARGS_8(A1, A2, A3, A4, A5, A6, A7, A8)                            \
  a1, a2, a3, a4, a5, a6, a7, a8
#define NBLA_ARGS_9(A1, A2, A3, A4, A5, A6, A7, A8, A9)                        \
  a1, a2, a3, a4, a5, a6, a7, a8, a9
#define NBLA_ARGS_10(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)                  \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10
#define NBLA_ARGS_11(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)             \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11
#define NBLA_ARGS_12(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)        \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12
#define NBLA_ARGS_13(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13)   \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13
#define NBLA_ARGS_14(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14)                                                      \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14
#define NBLA_ARGS_15(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15)                                                 \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
#define NBLA_ARGS_16(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16)                                            \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16
#define NBLA_ARGS_17(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17)                                       \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17

#define NBLA_ARG_N(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, \
                   _14, _15, _16, _17, N, ...)                                 \
  N
#ifdef _MSC_VER
#define NBLA_NUM_ARGS_(...)                                                    \
  NBLA_EXPAND(NBLA_ARG_N(__VA_ARGS__, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, \
                         6, 5, 4, 3, 2, 1, 0))
#else
#define NBLA_NUM_ARGS_(...)                                                    \
  NBLA_ARG_N(__VA_ARGS__, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, \
             2, 1, 0)
#endif

#define NBLA_MACRO_OVERRIDE(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11,  \
                            _12, _13, _14, _15, _16, _17, NAME, ...)           \
  NAME
// Helper end-------------------------------------------------------------------

#ifdef _MSC_VER
/**
Get number of args of variadic macro

ex) NBLA_NUM_ARGS(foo, bar) --> 2
*/
#define NBLA_NUM_ARGS(...) NBLA_EXPAND(NBLA_NUM_ARGS_(0, ##__VA_ARGS__))

/**
Convert type arguments to definitions

ex) NBLA_ARGDEFS(int, const string&) --> int a1, const string& a2
*/
#define NBLA_ARGDEFS(...)                                                      \
  NBLA_EXPAND(NBLA_EXPAND(NBLA_MACRO_OVERRIDE(                                 \
      0, ##__VA_ARGS__, NBLA_ARGDEFS_17, NBLA_ARGDEFS_16, NBLA_ARGDEFS_15,     \
      NBLA_ARGDEFS_14, NBLA_ARGDEFS_13, NBLA_ARGDEFS_12, NBLA_ARGDEFS_11,      \
      NBLA_ARGDEFS_10, NBLA_ARGDEFS_9, NBLA_ARGDEFS_8, NBLA_ARGDEFS_7,         \
      NBLA_ARGDEFS_6, NBLA_ARGDEFS_5, NBLA_ARGDEFS_4, NBLA_ARGDEFS_3,          \
      NBLA_ARGDEFS_2, NBLA_ARGDEFS_1, NBLA_ARGDEFS_0))(__VA_ARGS__))

/**
Convert type arguments to argument names

ex) NBLA_ARGS(int, const string&) --> a1, a2
*/
#define NBLA_ARGS(...)                                                         \
  NBLA_EXPAND(NBLA_EXPAND(NBLA_MACRO_OVERRIDE(                                 \
      0, ##__VA_ARGS__, NBLA_ARGS_17, NBLA_ARGS_16, NBLA_ARGS_15,              \
      NBLA_ARGS_14, NBLA_ARGS_13, NBLA_ARGS_12, NBLA_ARGS_11, NBLA_ARGS_10,    \
      NBLA_ARGS_9, NBLA_ARGS_8, NBLA_ARGS_7, NBLA_ARGS_6, NBLA_ARGS_5,         \
      NBLA_ARGS_4, NBLA_ARGS_3, NBLA_ARGS_2, NBLA_ARGS_1,                      \
      NBLA_ARGS_0))(__VA_ARGS__))
#else
#define NBLA_NUM_ARGS(...) NBLA_NUM_ARGS_(0, ##__VA_ARGS__)
#define NBLA_ARGDEFS(...)                                                      \
  NBLA_MACRO_OVERRIDE(                                                         \
      0, ##__VA_ARGS__, NBLA_ARGDEFS_17, NBLA_ARGDEFS_16, NBLA_ARGDEFS_15,     \
      NBLA_ARGDEFS_14, NBLA_ARGDEFS_13, NBLA_ARGDEFS_12, NBLA_ARGDEFS_11,      \
      NBLA_ARGDEFS_10, NBLA_ARGDEFS_9, NBLA_ARGDEFS_8, NBLA_ARGDEFS_7,         \
      NBLA_ARGDEFS_6, NBLA_ARGDEFS_5, NBLA_ARGDEFS_4, NBLA_ARGDEFS_3,          \
      NBLA_ARGDEFS_2, NBLA_ARGDEFS_1, NBLA_ARGDEFS_0)                          \
  (__VA_ARGS__)
#define NBLA_ARGS(...)                                                         \
  NBLA_MACRO_OVERRIDE(0, ##__VA_ARGS__, NBLA_ARGS_17, NBLA_ARGS_16,            \
                      NBLA_ARGS_15, NBLA_ARGS_14, NBLA_ARGS_13, NBLA_ARGS_12,  \
                      NBLA_ARGS_11, NBLA_ARGS_10, NBLA_ARGS_9, NBLA_ARGS_8,    \
                      NBLA_ARGS_7, NBLA_ARGS_6, NBLA_ARGS_5, NBLA_ARGS_4,      \
                      NBLA_ARGS_3, NBLA_ARGS_2, NBLA_ARGS_1, NBLA_ARGS_0)      \
  (__VA_ARGS__)
#endif
#endif
