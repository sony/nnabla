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

/** Exception throwing utilities.
*/
#ifndef __NBLA_EXCEPTION_HPP__
#define __NBLA_EXCEPTION_HPP__

#if defined(_MSC_VER)
///< definition of __func__
#define __func__ __FUNCTION__
#endif

#include <nbla/defs.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nbla {
using std::string;
using std::snprintf;
using std::vector;

/** In NNabla, exceptions are thrown through this macro. Error codes are
    defined in enum class nbla::error_code. See also NBLA_CHECK.

    Example: `NBLA_ERROR(error_code::cuda_error, "Error size %d", size);`
*/
#define NBLA_ERROR(code, msg, ...)                                             \
  throw Exception(code, format_string(msg, ##__VA_ARGS__), __func__, __FILE__, \
                  __LINE__);

/** You can check whether the specified condition is met, or raise error with
    specified message.

    Example: `NBLA_CHECK(size == 2,
                         error_code::cuda_error, "Error size %d", size);`
*/
#define NBLA_CHECK(condition, code, msg, ...)                                  \
  if (!(condition)) {                                                          \
    NBLA_ERROR(code, string("Failed `" #condition "`: ") + msg,                \
               ##__VA_ARGS__);                                                 \
  }

#define NBLA_FORCE_ASSERT(condition, msg, ...)                                 \
  if (!(condition)) {                                                          \
    std::cerr << "Aborting: " << format_string(msg, ##__VA_ARGS__) << " at "   \
              << __func__ << " in " << __FILE__ << ":" << __LINE__             \
              << std::endl;                                                    \
    ::abort();                                                                 \
  }

/** Enum of error codes throwing in NNabla

Note: Developers must add a line `CASE_ERROR_STRING({code name});` into
get_error_string function in src/nbla/exception.cpp, if a new code is added.
*/
enum class error_code {
  unclassified,
  not_implemented,
  value,
  type,
  memory,
  io,
  os,
  target_specific,
  target_specific_async,
  runtime
};

string get_error_string(error_code code);

/** Exception class of NNabla

Error codes are enumerated in enum class error_code you can find above.
It is not expected that developers/users throw this exception directly.
Instead, use NBLA_ERROR macro.
*/
// https://github.com/Itseez/opencv/blob/c3ad8af42a85a3d03c6dd5727c8b5f4f7585d1d2/modules/core/src/system.cpp
// https://github.com/Itseez/opencv/blob/9aeb8c8d5a35bf7ed5208459d46fdb6822c5692c/modules/core/include/opencv2/core/base.hpp
// https://github.com/Itseez/opencv/blob/b2d44663fdd90e4c50d4a06435492b5cb0f1021d/modules/core/include/opencv2/core.hpp
class NBLA_API Exception : public std::exception {
protected:
  error_code code_; ///< error code
  string full_msg_; ///< Buffer of full message to be shown
  string msg_;      ///< error message
  string func_;     ///< function name
  string file_;     ///< file name
  int line_;        ///< line no.
public:
  Exception(error_code code, const string &msg, const string &func,
            const string &file, int line);
  virtual ~Exception() throw();
  virtual const char *what() const throw();
};

/** String formatter.
*/
template <typename T, typename... Args>
string format_string(const string &format, T first, Args... rest) {
  int size = snprintf(nullptr, 0, format.c_str(), first, rest...);
  if (size < 0) {
    std::printf("fatal error in format_string function: snprintf failed\n");
    std::abort();
  }
  vector<char> buffer(size + 1);
  snprintf(buffer.data(), size + 1, format.c_str(), first, rest...);
  return string(buffer.data(), buffer.data() + size);
}

/** String formatter without format.
*/
inline string format_string(const string &format) {
  for (auto itr = format.begin(); itr != format.end(); itr++) {
    if (*itr == '%') {
      if (*(itr + 1) == '%') {
        itr++;
      } else {
        NBLA_ERROR(error_code::unclassified, "Invalid format string %s",
                   format.c_str());
      }
    }
  }
  return format;
}
}

#endif
