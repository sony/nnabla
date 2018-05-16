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

// -*- coding:utf-8 -*-
#include <nbla/logger.hpp>

#include <string>

#ifdef _WIN32
#include <cstdio>
#include <cstdlib>
#include <direct.h>
#include <shlobj.h>
#include <windows.h>
#else
#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

std::shared_ptr<spdlog::logger> get_logger(void) {
  static std::shared_ptr<spdlog::logger> l = 0;
  if (l == 0) {
    std::string logpath;
    std::string logfile;
#ifdef _WIN32
    TCHAR szPath[MAX_PATH];

    if (SUCCEEDED(SHGetFolderPath(nullptr, CSIDL_APPDATA | CSIDL_FLAG_CREATE, nullptr,
                                  0, szPath))) {
      logpath = szPath;
      logpath += "\\NNabla";
      _mkdir(logpath.c_str());
      logpath += "\\log";
      _mkdir(logpath.c_str());
      logfile = logpath + "\\nbla_lib.log";
    }
#else
    const char *homedir = getenv("HOME");
    if (homedir == nullptr) {
      struct passwd *pw = getpwuid(getuid());
      if (pw != nullptr) {
        homedir = pw->pw_dir;
      }
    }
    if (homedir == nullptr) {
      logpath = "/tmp_";
      logpath += getuid();
    } else {
      logpath = homedir;
    }
    logpath += "/nnabla_data";
    mkdir(logpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    logpath += "/log";
    mkdir(logpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    logfile = logpath + "/nbla_lib.log";
#endif

    std::vector<spdlog::sink_ptr> sinks;
    auto s1 = std::make_shared<spdlog::sinks::stdout_sink_st>();
    s1->set_level(spdlog::level::critical);
    sinks.push_back(s1);

    auto s2 = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        logfile.c_str(), 1024 * 1024, 5);
    sinks.push_back(s2);

#ifdef SPDLOG_ENABLE_SYSLOG
    auto s3 = std::make_shared<spdlog::sinks::syslog_sink>();
    sinks.push_back(s3);
#endif
    l = std::make_shared<spdlog::logger>("nbla", begin(sinks), end(sinks));
  }
  return l;
}
