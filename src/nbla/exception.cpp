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

#include <nbla/exception.hpp>

#include <sstream>

namespace nbla {

// Get error string from enum class error_code.
string get_error_string(error_code code) {
  switch (code) {
#define CASE_ERROR_STRING(code_name)                                           \
  case error_code::code_name:                                                  \
    return #code_name;

    CASE_ERROR_STRING(unclassified);
    CASE_ERROR_STRING(not_implemented);
    CASE_ERROR_STRING(value);
    CASE_ERROR_STRING(type);
    CASE_ERROR_STRING(memory);
    CASE_ERROR_STRING(io);
    CASE_ERROR_STRING(os);
    CASE_ERROR_STRING(target_specific);
    CASE_ERROR_STRING(target_specific_async);
    CASE_ERROR_STRING(runtime);
#undef CASE_ERROR_STRING
  }
  return std::string();
}

// -------------------------------------------------------------------------
// Define Exception class impl
// -------------------------------------------------------------------------
Exception::Exception(error_code code, const string &msg, const string &func,
                     const string &file, int line)
    : code_(code), msg_(msg), func_(func), file_(file), line_(line) {
  std::ostringstream ss;
  ss << get_error_string(code_) << " error in " << func_ << std::endl
     << file_ << ":" << line_ << std::endl
     << msg_ << std::endl;
  full_msg_ = ss.str();
}
Exception::~Exception() throw() {}

const char *Exception::what() const throw() { return full_msg_.c_str(); }
// -------------------------------------------------------------------------
// END: Define Exception class impl
// -------------------------------------------------------------------------
}
