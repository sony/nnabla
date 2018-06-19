:: Copyright (c) 2017 Sony Corporation. All Rights Reserved.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

SET protobuf_tag=v3.4.1

:: Build protobuf libs
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
IF NOT EXIST %protobuf_folder% (
	git clone https://github.com/google/protobuf.git --branch %protobuf_tag% --depth=1 %protobuf_folder% || GOTO :error
)

CD %protobuf_folder%
IF NOT EXIST build-folder (
   MD build-folder
)

CD build-folder
cmake.exe -G "%generate_target%" -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_TESTS=OFF ..\cmake || GOTO :error
cmake.exe --build . --config %build_type% || GOTO :error
