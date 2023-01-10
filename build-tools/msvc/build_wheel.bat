REM Copyright 2018,2019,2020,2021 Sony Corporation.
REM Copyright 2021 Sony Group Corporation.
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
REM                PYTHON_VERSION: 3.7, 3.8, 3.9 or 3.10
REM     (optional) VISUAL_STUDIO_ EDITION: 2015 or 2019(experimental)
REM
SETLOCAL

SET SKIP_VC_SETUP=True
CALL %~dp0tools\env.bat %1 2019|| GOTO :error

@ECHO ON

REM Build Wheel
IF NOT EXIST %nnabla_build_wheel_folder%\ MD %nnabla_build_wheel_folder%
CD %nnabla_build_wheel_folder%

SET third_party_folder=%nnabla_root%\third_party
CALL %~dp0tools\build_protobuf.bat   || GOTO :error

REM Download bulit flatbuffers binary
CALL %~dp0tools\get_flatbuffers.bat || GOTO :error

CD %nnabla_build_wheel_folder%

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

cmake --build . --config %build_type% --target wheel || GOTO :error

GOTO :end
:error
ECHO failed with error code %errorlevel%.
ENDLOCAL
exit /b %errorlevel%

:end
ENDLOCAL
