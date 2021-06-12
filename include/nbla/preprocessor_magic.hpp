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
#define NBLA_ARGDEFS_18(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18)                          \
  NBLA_ARGDEFS_17(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17)                                               \
  , A18 a18
#define NBLA_ARGDEFS_19(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19)                     \
  NBLA_ARGDEFS_18(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18)                                          \
  , A19 a19
#define NBLA_ARGDEFS_20(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20)                \
  NBLA_ARGDEFS_19(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19)                                     \
  , A20 a20
#define NBLA_ARGDEFS_21(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21)           \
  NBLA_ARGDEFS_20(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20)                                \
  , A21 a21
#define NBLA_ARGDEFS_22(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21, A22)      \
  NBLA_ARGDEFS_21(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20, A21)                           \
  , A22 a22
#define NBLA_ARGDEFS_23(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23) \
  NBLA_ARGDEFS_22(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20, A21, A22)                      \
  , A23 a23
#define NBLA_ARGDEFS_24(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, \
                        A24)                                                   \
  NBLA_ARGDEFS_23(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20, A21, A22, A23)                 \
  , A24 a24
#define NBLA_ARGDEFS_25(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, \
                        A24, A25)                                              \
  NBLA_ARGDEFS_24(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20, A21, A22, A23, A24)            \
  , A25 a25
#define NBLA_ARGDEFS_26(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, \
                        A24, A25, A26)                                         \
  NBLA_ARGDEFS_25(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25)       \
  , A26 a26
#define NBLA_ARGDEFS_27(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, \
                        A24, A25, A26, A27)                                    \
  NBLA_ARGDEFS_26(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26)  \
  , A27 a27
#define NBLA_ARGDEFS_28(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, \
                        A24, A25, A26, A27, A28)                               \
  NBLA_ARGDEFS_27(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26,  \
                  A27)                                                         \
  , A28 a28
#define NBLA_ARGDEFS_29(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, \
                        A24, A25, A26, A27, A28, A29)                          \
  NBLA_ARGDEFS_28(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26,  \
                  A27, A28)                                                    \
  , A29 a29
#define NBLA_ARGDEFS_30(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, \
                        A24, A25, A26, A27, A28, A29, A30)                     \
  NBLA_ARGDEFS_29(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26,  \
                  A27, A28, A29)                                               \
  , A30 a30
#define NBLA_ARGDEFS_31(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, \
                        A24, A25, A26, A27, A28, A29, A30, A31)                \
  NBLA_ARGDEFS_30(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26,  \
                  A27, A28, A29, A30)                                          \
  , A31 a31
#define NBLA_ARGDEFS_32(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,     \
                        A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, \
                        A24, A25, A26, A27, A28, A29, A30, A31, A32)           \
  NBLA_ARGDEFS_31(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, \
                  A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26,  \
                  A27, A28, A29, A30, A31)                                     \
  , A32 a32

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
#define NBLA_ARGS_18(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18)                                  \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18
#define NBLA_ARGS_19(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19)                             \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19
#define NBLA_ARGS_20(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20)                        \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20
#define NBLA_ARGS_21(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21)                   \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21
#define NBLA_ARGS_22(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21, A22)              \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21, a22
#define NBLA_ARGS_23(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21, A22, A23)         \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21, a22, a23
#define NBLA_ARGS_24(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24)    \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21, a22, a23, a24
#define NBLA_ARGS_25(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24,    \
                     A25)                                                      \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21, a22, a23, a24, a25
#define NBLA_ARGS_26(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24,    \
                     A25, A26)                                                 \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21, a22, a23, a24, a25, a26
#define NBLA_ARGS_27(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24,    \
                     A25, A26, A27)                                            \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21, a22, a23, a24, a25, a26, a27
#define NBLA_ARGS_28(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24,    \
                     A25, A26, A27, A28)                                       \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28
#define NBLA_ARGS_29(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24,    \
                     A25, A26, A27, A28, A29)                                  \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29
#define NBLA_ARGS_30(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24,    \
                     A25, A26, A27, A28, A29, A30)                             \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30
#define NBLA_ARGS_31(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24,    \
                     A25, A26, A27, A28, A29, A30, A31)                        \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31
#define NBLA_ARGS_32(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,   \
                     A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24,    \
                     A25, A26, A27, A28, A29, A30, A31, A32)                   \
  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,  \
      a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,    \
      a32

#define NBLA_ARG_N(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, \
                   _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, \
                   _26, _27, _28, _29, _30, _31, _32, N, ...)                  \
  N
#ifdef _MSC_VER
#define NBLA_NUM_ARGS_(...)                                                    \
  NBLA_EXPAND(NBLA_ARG_N(__VA_ARGS__, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,  \
                         22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,   \
                         9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

#else
#define NBLA_NUM_ARGS_(...)                                                    \
  NBLA_ARG_N(__VA_ARGS__, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,  \
             19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,   \
             1, 0)
#endif

#define NBLA_MACRO_OVERRIDE(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11,  \
                            _12, _13, _14, _15, _16, _17, _18, _19, _20, _21,  \
                            _22, _23, _24, _25, _26, _27, _28, _29, _30, _31,  \
                            _32, NAME, ...)                                    \
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
      0, ##__VA_ARGS__, NBLA_ARGDEFS_32, NBLA_ARGDEFS_31, NBLA_ARGDEFS_30,     \
      NBLA_ARGDEFS_29, NBLA_ARGDEFS_28, NBLA_ARGDEFS_27, NBLA_ARGDEFS_26,      \
      NBLA_ARGDEFS_25, NBLA_ARGDEFS_24, NBLA_ARGDEFS_23, NBLA_ARGDEFS_22,      \
      NBLA_ARGDEFS_21, NBLA_ARGDEFS_20, NBLA_ARGDEFS_19, NBLA_ARGDEFS_18,      \
      NBLA_ARGDEFS_17, NBLA_ARGDEFS_16, NBLA_ARGDEFS_15, NBLA_ARGDEFS_14,      \
      NBLA_ARGDEFS_13, NBLA_ARGDEFS_12, NBLA_ARGDEFS_11, NBLA_ARGDEFS_10,      \
      NBLA_ARGDEFS_9, NBLA_ARGDEFS_8, NBLA_ARGDEFS_7, NBLA_ARGDEFS_6,          \
      NBLA_ARGDEFS_5, NBLA_ARGDEFS_4, NBLA_ARGDEFS_3, NBLA_ARGDEFS_2,          \
      NBLA_ARGDEFS_1, NBLA_ARGDEFS_0))(__VA_ARGS__))

/**
Convert type arguments to argument names

ex) NBLA_ARGS(int, const string&) --> a1, a2
*/
#define NBLA_ARGS(...)                                                         \
  NBLA_EXPAND(NBLA_EXPAND(NBLA_MACRO_OVERRIDE(                                 \
      0, ##__VA_ARGS__, NBLA_ARGS_32, NBLA_ARGS_31, NBLA_ARGS_30,              \
      NBLA_ARGS_29, NBLA_ARGS_28, NBLA_ARGS_27, NBLA_ARGS_26, NBLA_ARGS_25,    \
      NBLA_ARGS_24, NBLA_ARGS_23, NBLA_ARGS_22, NBLA_ARGS_21, NBLA_ARGS_20,    \
      NBLA_ARGS_19, NBLA_ARGS_18, NBLA_ARGS_17, NBLA_ARGS_16, NBLA_ARGS_15,    \
      NBLA_ARGS_14, NBLA_ARGS_13, NBLA_ARGS_12, NBLA_ARGS_11, NBLA_ARGS_10,    \
      NBLA_ARGS_9, NBLA_ARGS_8, NBLA_ARGS_7, NBLA_ARGS_6, NBLA_ARGS_5,         \
      NBLA_ARGS_4, NBLA_ARGS_3, NBLA_ARGS_2, NBLA_ARGS_1,                      \
      NBLA_ARGS_0))(__VA_ARGS__))
#else
#define NBLA_NUM_ARGS(...) NBLA_NUM_ARGS_(0, ##__VA_ARGS__)
#define NBLA_ARGDEFS(...)                                                      \
  NBLA_MACRO_OVERRIDE(                                                         \
      0, ##__VA_ARGS__, NBLA_ARGDEFS_32, NBLA_ARGDEFS_31, NBLA_ARGDEFS_30,     \
      NBLA_ARGDEFS_29, NBLA_ARGDEFS_28, NBLA_ARGDEFS_27, NBLA_ARGDEFS_26,      \
      NBLA_ARGDEFS_25, NBLA_ARGDEFS_24, NBLA_ARGDEFS_23, NBLA_ARGDEFS_22,      \
      NBLA_ARGDEFS_21, NBLA_ARGDEFS_20, NBLA_ARGDEFS_19, NBLA_ARGDEFS_18,      \
      NBLA_ARGDEFS_17, NBLA_ARGDEFS_16, NBLA_ARGDEFS_15, NBLA_ARGDEFS_14,      \
      NBLA_ARGDEFS_13, NBLA_ARGDEFS_12, NBLA_ARGDEFS_11, NBLA_ARGDEFS_10,      \
      NBLA_ARGDEFS_9, NBLA_ARGDEFS_8, NBLA_ARGDEFS_7, NBLA_ARGDEFS_6,          \
      NBLA_ARGDEFS_5, NBLA_ARGDEFS_4, NBLA_ARGDEFS_3, NBLA_ARGDEFS_2,          \
      NBLA_ARGDEFS_1, NBLA_ARGDEFS_0)                                          \
  (__VA_ARGS__)
#define NBLA_ARGS(...)                                                         \
  NBLA_MACRO_OVERRIDE(                                                         \
      0, ##__VA_ARGS__, NBLA_ARGS_32, NBLA_ARGS_31, NBLA_ARGS_30,              \
      NBLA_ARGS_29, NBLA_ARGS_28, NBLA_ARGS_27, NBLA_ARGS_26, NBLA_ARGS_25,    \
      NBLA_ARGS_24, NBLA_ARGS_23, NBLA_ARGS_22, NBLA_ARGS_21, NBLA_ARGS_20,    \
      NBLA_ARGS_19, NBLA_ARGS_18, NBLA_ARGS_17, NBLA_ARGS_16, NBLA_ARGS_15,    \
      NBLA_ARGS_14, NBLA_ARGS_13, NBLA_ARGS_12, NBLA_ARGS_11, NBLA_ARGS_10,    \
      NBLA_ARGS_9, NBLA_ARGS_8, NBLA_ARGS_7, NBLA_ARGS_6, NBLA_ARGS_5,         \
      NBLA_ARGS_4, NBLA_ARGS_3, NBLA_ARGS_2, NBLA_ARGS_1, NBLA_ARGS_0)         \
  (__VA_ARGS__)
#endif
#endif
