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
REM To build with Debug mode, set environment variable `build_type` as `Debug` as following
REM before running this script.
REM set build_type=Debug

REM 
REM Usage:
REM   build_wheel.bat PYTHON_VERSION [VISUAL_STUDIO_ EDITION]
REM                PYTHON_VERSION: 3.6, 3.7 or 3.8
REM     (optional) VISUAL_STUDIO_ EDITION: 2015 or 2019(experimental)
REM
SETLOCAL

REM Environment
CALL %~dp0tools\env.bat %1 || GOTO :error

REM Build Wheel
IF NOT EXIST %nnabla_build_wheel_folder%\ MD %nnabla_build_wheel_folder%
CD %nnabla_build_wheel_folder%

SET third_party_folder=%nnabla_root%\third_party
CALL %~dp0tools\build_protobuf.bat   || GOTO :error

cmake -G "%generate_target%" ^
      -DBUILD_CPP_LIB=OFF ^
      -DBUILD_PYTHON_PACKAGE=ON ^
      -DCPPLIB_BUILD_DIR=%nnabla_build_folder% ^
      -DCPPLIB_LIBRARY=%nnabla_build_folder%\bin\%build_type%\nnabla%lib_name_suffix%.dll ^
      -DLIB_NAME_SUFFIX=%lib_name_suffix% ^
      -DPYTHON_COMMAND_NAME=python ^
      -DWHEEL_SUFFIX=%wheel_suffix% ^
      -DPROTOC_COMMAND=%protobuf_protoc_executable% ^
      %nnabla_root% || GOTO :error

msbuild wheel.vcxproj /p:Configuration=%build_type% || GOTO :error

GOTO :end
:error
ECHO failed with error code %errorlevel%.
ENDLOCAL
exit /b %errorlevel%

:end
ENDLOCAL
