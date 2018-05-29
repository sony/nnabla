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

:: To build with Debug mode, set environment variable `build_type` as `Debug` as following
:: before running this script.
:: set build_type=Debug

@ECHO OFF

SETLOCAL

:: Settings
CALL %~dp0tools\default_settings.bat || GOTO :error

:: Folders
CALL %~dp0tools\default_folders.bat || GOTO :error

:: Build Wheel
IF NOT EXIST %nnabla_build_wheel_folder%\ MD %nnabla_build_wheel_folder%
CD %nnabla_build_wheel_folder%

cmake -G "%generate_target%" ^
      -DPYTHON_COMMAND_NAME=python ^
      -DBUILD_CPP_LIB=OFF ^
      -DBUILD_PYTHON_PACKAGE=ON ^
      -DCPPLIB_BUILD_DIR=%nnabla_build_folder% ^
      -DCPPLIB_LIBRARY=%nnabla_build_folder%\bin\%build_type%\nnabla.dll ^
      %nnabla_root% || GOTO :error

msbuild wheel.vcxproj /p:Configuration=%build_type% || GOTO :error

ENDLOCAL
exit /b

:error
ECHO failed with error code %errorlevel%.
ENDLOCAL
exit /b %errorlevel%

