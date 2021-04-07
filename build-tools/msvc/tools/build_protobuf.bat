@ECHO OFF

REM Copyright (c) 2017 Sony Corporation. All Rights Reserved.
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.

for /f %%i in ('python -c "import google.protobuf; print(google.protobuf.__version__)"') do set PROTOVER=%%i
SET protobuf_tag=v%PROTOVER%

REM Build protobuf libs
SET protobuf_folder=%third_party_folder%\protobuf-%protobuf_tag%
SET protobuf_include_dir=%protobuf_folder%\src
SET protobuf_bin_folder=%protobuf_folder%\build-folder\%build_type%
SET protobuf_lib_suffix=.lib
IF [%build_type%] == [Debug] (
  SET protobuf_lib_suffix=d.lib
)
SET protobuf_library=%protobuf_bin_folder%\libprotobuf%protobuf_lib_suffix%
SET protobuf_lite_library=%protobuf_bin_folder%\libprotobuf-lite%protobuf_lib_suffix%
SET protobuf_protoc_executable=%protobuf_bin_folder%\protoc.exe

IF EXIST %protobuf_library% (
   ECHO libprotobuf already exists. Skipping...
   EXIT /b
)

IF NOT EXIST %protobuf_folder% (
   git clone -c core.longpaths=true https://github.com/protocolbuffers/protobuf.git --branch %protobuf_tag% --depth=1 %protobuf_folder% || GOTO :error
)

CD %protobuf_folder%
IF NOT EXIST build-folder (
   MD build-folder
)

CD build-folder
cmake.exe -G "%generate_target%" -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_TESTS=OFF ..\cmake || GOTO :error
cmake.exe --build . --config %build_type% || GOTO :error


:error
ECHO failed with error code %errorlevel%.

exit /b %errorlevel%
