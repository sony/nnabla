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
:: 

@ECHO OFF
SETLOCAL

:: Settings
CALL %~dp0tools\default_settings.bat || GOTO :error

:: Folders
CALL %~dp0tools\default_folders.bat || GOTO :error

SET third_party_folder=%nnabla_root%\third_party

:: Build third party libraries.
CALL %~dp0tools\build_zlib.bat       || GOTO :error
CALL %~dp0tools\build_libarchive.bat || GOTO :error
CALL %~dp0tools\build_protobuf.bat   || GOTO :error

:: Build CPP library.
ECHO "--- CMake options for C++ build ---"
set nnabla_debug_options=
IF [%build_type%] == [Debug] (
  set nnabla_debug_options=-DCMAKE_CXX_FLAGS="/bigobj"
)

IF NOT EXIST %nnabla_build_folder% (
   MD %nnabla_build_folder%
)

IF NOT EXIST %nnabla_build_folder%\bin (
   MD %nnabla_build_folder%\bin
)

IF NOT EXIST %nnabla_build_folder%\bin\%build_type% ^(
    MD %nnabla_build_folder%\bin\%build_type%
)

powershell ^"Get-ChildItem %third_party_folder% -Filter *.dll -Recurse ^| ForEach-Object {Copy-Item $_.FullName %nnabla_build_folder%\bin\%build_type%}^"
powershell ^"Get-ChildItem %third_party_folder% -Filter *.lib -Recurse ^| ForEach-Object {Copy-Item $_.FullName %nnabla_build_folder%\bin\%build_type%}^"
powershell ^"Get-ChildItem %third_party_folder% -Filter *.exp -Recurse ^| ForEach-Object {Copy-Item $_.FullName %nnabla_build_folder%\bin\%build_type%}^"

CD %nnabla_build_folder%

ECHO OFF
cmake -G "%generate_target%" ^
      -DPYTHON_COMMAND_NAME=python ^
      -DBUILD_CPP_UTILS=ON ^
      -DBUILD_PYTHON_PACKAGE=OFF ^
      -DLibArchive_LIBRARY=%libarchive_library% ^
      -DLibArchive_INCLUDE_DIR=%libarchive_include_dir% ^
      -DProtobuf_INCLUDE_DIR=%protobuf_include_dir% ^
      -DProtobuf_LIBRARY=%protobuf_library% ^
      -DProtobuf_LITE_LIBRARY=%protobuf_lite_library% ^
      -DProtobuf_PROTOC_EXECUTABLE=%protobuf_protoc_executable% ^
      -DPROTOC_COMMAND=%protobuf_protoc_executable% ^
      -DZLIB_LIBRARY_RELEASE=%zlib_library% ^
      -DZLIB_INCLUDE_DIR=%zlib_include_dir% ^
      %nnabla_debug_options% ^
      %nnabla_root% || GOTO :error

cmake --build . --config %build_type% || GOTO :error

ENDLOCAL
exit /b

:error
ECHO failed with error code %errorlevel%.

ENDLOCAL
exit /b %errorlevel%

