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
REM 

REM 
REM Usage:
REM   build_cpplib.bat [VISUAL_STUDIO_ EDITION]
REM     (optional) VISUAL_STUDIO_ EDITION: 2015 or 2019(experimental)
REM

SETLOCAL

REM Environment
CALL %~dp0tools\env.bat 3.6 %1 || GOTO :error
SET third_party_folder=%nnabla_root%\third_party

REM Build third party libraries.
CALL %~dp0tools\build_zlib.bat       || GOTO :error
CALL %~dp0tools\build_libarchive.bat || GOTO :error
CALL %~dp0tools\build_protobuf.bat   || GOTO :error

REM Get pre-built lz4 and zstd libraries
CALL %~dp0tools\get_liblz4.bat || GOTO :error
CALL %~dp0tools\get_libzstd.bat || GOTO :error

REM Build CPP library.
ECHO "--- CMake options for C++ build ---"
set nnabla_debug_options=
IF [%build_type%] == [Debug] SET nnabla_debug_options=-DCMAKE_CXX_FLAGS="/bigobj"

IF NOT EXIST %nnabla_build_folder% MD %nnabla_build_folder%
IF NOT EXIST %nnabla_build_folder%\bin MD %nnabla_build_folder%\bin
IF NOT EXIST %nnabla_build_folder%\bin\%build_type% MD %nnabla_build_folder%\bin\%build_type%

powershell ^"Get-ChildItem %third_party_folder% -Filter *.dll -Recurse ^| ForEach-Object {Copy-Item $_.FullName %nnabla_build_folder%\bin\%build_type%}^"
powershell ^"Get-ChildItem %third_party_folder% -Filter *.lib -Recurse ^| ForEach-Object {Copy-Item $_.FullName %nnabla_build_folder%\bin\%build_type%}^"
powershell ^"Get-ChildItem %third_party_folder% -Filter *.exp -Recurse ^| ForEach-Object {Copy-Item $_.FullName %nnabla_build_folder%\bin\%build_type%}^"

CD %nnabla_build_folder%

cmake -G "%generate_target%" ^
      -DBUILD_CPP_UTILS=ON ^
      -DBUILD_TEST=ON ^
      -DBUILD_PYTHON_PACKAGE=OFF ^
      -Dgtest_force_shared_crt=TRUE ^
      -DLIB_NAME_SUFFIX=%lib_name_suffix% ^
      -DLibArchive_INCLUDE_DIR=%libarchive_include_dir% ^
      -DLibArchive_LIBRARY=%libarchive_library% ^
      -DPROTOC_COMMAND=%protobuf_protoc_executable% ^
      -DPYTHON_COMMAND_NAME=python ^
      -DProtobuf_INCLUDE_DIR=%protobuf_include_dir% ^
      -DProtobuf_LIBRARY=%protobuf_library% ^
      -DProtobuf_LITE_LIBRARY=%protobuf_lite_library% ^
      -DProtobuf_PROTOC_EXECUTABLE=%protobuf_protoc_executable% ^
      -DZLIB_INCLUDE_DIR=%zlib_include_dir% ^
      -DZLIB_LIBRARY_RELEASE=%zlib_library% ^
      %nnabla_debug_options% ^
      %nnabla_root% || GOTO :error

cmake --build . --config %build_type% || GOTO :error
cmake --build . --config %build_type% --target test_nbla_utils || GOTO :error
SET PATH="%ProgramFiles%\Cmake\bin";%PATH%
cpack -G ZIP -C %build_type%

GOTO :end
:error
ECHO failed with error code %errorlevel%.
exit /b %errorlevel%

:end
ENDLOCAL
