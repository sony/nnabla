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

#ifndef __NBLA_UTILS_EIGEN_HPP__
#define __NBLA_UTILS_EIGEN_HPP__

#define EIGEN_NO_DEBUG
#define EIGEN_MPL2_ONLY

#include <Eigen/Dense>

namespace nbla {

namespace eigen {
template <typename T>
using Matrix =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T> using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>;

template <typename T> using ColVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T> using MatrixMap = Eigen::Map<Matrix<T>>;

template <typename T> using ConstMatrixMap = Eigen::Map<const Matrix<T>>;

template <typename T> using RowVectorMap = Eigen::Map<RowVector<T>>;

template <typename T> using ConstRowVectorMap = Eigen::Map<const RowVector<T>>;

template <typename T> using ColVectorMap = Eigen::Map<ColVector<T>>;

template <typename T> using ConstColVectorMap = Eigen::Map<const ColVector<T>>;

// C OP A * B with all combination of transpose or non-transpose
// OP is for example '=' and '+='.
#define NBLA_EIGEN_MATMUL_T(C, Tc, A, Ta, B, Tb, OP)                           \
  if (Tc) {                                                                    \
    if (Ta) {                                                                  \
      if (Tb) {                                                                \
        C.transpose() OP A.transpose() * B.transpose();                        \
      } else {                                                                 \
        C.transpose() OP A.transpose() * B;                                    \
      }                                                                        \
    } else {                                                                   \
      if (Tb) {                                                                \
        C.transpose() OP A *B.transpose();                                     \
      } else {                                                                 \
        C.transpose() OP A *B;                                                 \
      }                                                                        \
    }                                                                          \
  } else {                                                                     \
    if (Ta) {                                                                  \
      if (Tb) {                                                                \
        C OP A.transpose() * B.transpose();                                    \
      } else {                                                                 \
        C OP A.transpose() * B;                                                \
      }                                                                        \
    } else {                                                                   \
      if (Tb) {                                                                \
        C OP A *B.transpose();                                                 \
      } else {                                                                 \
        C OP A *B;                                                             \
      }                                                                        \
    }                                                                          \
  }
}
}
#endif
