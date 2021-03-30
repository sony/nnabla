@ECHO OFF
REM Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
REM Install chocolatey (https://chocolatey.org/)
REM cinst cmake git
REM :for VC2015
REM cinst VisualCppBuildTools
REM :for VC2019
REM cinst visualstudio2019-workload-vctools
REM

CALL %~dp0python_env.bat %1

SET VCVER=%2
IF [%VCVER%] == [] (
   SET VCVER=2015
)

REM Visual Studio 2015 Compiler
IF NOT [%VCVER%] == [2015] GOTO :SKIP_VS2015
CALL "%ProgramFiles(x86)%\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
cl >NUL 2>NUL && GOTO :CL2015_FOUND
ECHO Visual Studio C++ compiler not found. Please check your installation.
EXIT /b 255
:CL2015_FOUND
msbuild -version >NUL 2>NUL && GOTO :MSBUILD2015_FOUND
ECHO msbuild not found. Please check your installation.
EXIT /b 255
:MSBUILD2015_FOUND

REM Visual Studio
IF DEFINED VS140COMNTOOLS GOTO :SKIP_VS140COMNTOOLS
SET VS140COMNTOOLS=%ProgramFiles(x86)%\Microsoft Visual Studio 14.0\Common7\Tools\
:SKIP_VS140COMNTOOLS
IF EXIST "%VS140COMNTOOLS%" GOTO :FINISH_VS140COMNTOOLS
ECHO VS140COMNTOOLS does not set properly.
EXIT /b 255
:FINISH_VS140COMNTOOLS
SET VS90COMNTOOLS=%VS140COMNTOOLS%
IF EXIST "%VS90COMNTOOLS%" GOTO :FINISH_VS90COMNTOOLS
ECHO VS90COMNTOOLS does not set properly.
EXIT /b 255
:FINISH_VS90COMNTOOLS

SET PATH=%ProgramFiles(x86)%\Windows Kits\10\bin\x64\ucrt;%PATH%
SET generate_target=Visual Studio 14 2015 Win64
:SKIP_VS2015

REM  Visual Studio 2019 Compiler
IF NOT [%VCVER%] == [2019] GOTO :SKIP_VS2019
CALL "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>NUL
cl >NUL 2>NUL && GOTO :CL2019_FOUND
ECHO Visual Studio C++ compiler not found. Please check your installation.
EXIT /b 255
:CL2019_FOUND
msbuild -version >NUL 2>NUL && GOTO :MSBUILD2019_FOUND
ECHO msbuild not found. Please check your installation.
EXIT /b 255
:MSBUILD2019_FOUND

SET generate_target=Visual Studio 16 2019
SET nnabla_build_folder_name=build_vs2019
:SKIP_VS2019


REM cmake
cmake --version >NUL 2>NUL && GOTO :CMAKE_FOUND
SET PATH="%ProgramFiles%\Cmake\bin";%PATH%
cmake --version >NUL 2>NUL && GOTO :CMAKE_FOUND
ECHO Please install cmake.
EXIT /b 255
:CMAKE_FOUND

REM git
git --version >NUL 2>NUL && GOTO :GIT_FOUND
SET PATH="%ProgramFiles%\Git\cmd";%PATH%
git --version >NUL 2>NUL && GOTO :GIT_FOUND
ECHO Please install git.
EXIT /b 255
:GIT_FOUND

SET nnabla_build_wheel_folder_suffix=_py%PYVER_MAJOR%%PYVER_MINOR%

SET nnabla_root=%~dp0..\..\..

IF NOT DEFINED nnabla_build_folder_name       SET nnabla_build_folder_name=build
IF NOT DEFINED nnabla_build_folder            SET nnabla_build_folder=%nnabla_root%\%nnabla_build_folder_name%
IF NOT DEFINED nnabla_build_wheel_folder_name SET nnabla_build_wheel_folder_name=build_wheel
IF NOT DEFINED nnabla_build_wheel_folder      SET nnabla_build_wheel_folder=%nnabla_root%\%nnabla_build_wheel_folder_name%%nnabla_build_wheel_folder_suffix%
IF NOT DEFINED nnabla_test_venv_folder        SET nnabla_test_venv_folder=%nnabla_build_wheel_folder%\env
IF NOT DEFINED generate_target                SET generate_target=Visual Studio 14 2015 Win64
IF NOT DEFINED build_type                     SET build_type=Release


