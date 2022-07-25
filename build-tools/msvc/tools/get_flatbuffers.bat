@ECHO OFF

REM Copyright 2022 Sony Group Corporation.
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

REM Download bulit flatbuffers binary

SET FLATC_PACKAGE=flatc-v2.0.0

call :download_flatc_and_rename https://github.com/google/flatbuffers/releases/download/v2.0.0/Windows.flatc.binary.zip flatc.exe flatc_windows.exe
call :download_flatc_and_rename https://github.com/google/flatbuffers/releases/download/v2.0.0/Mac.flatc.binary.zip flatc flatc_mac
call :download_flatc_and_rename https://github.com/google/flatbuffers/releases/download/v2.0.0/Linux.flatc.binary.clang++-9.zip flatc flatc_linux
exit /b

:download_flatc_and_rename
    powershell "[Net.ServicePointManager]::SecurityProtocol +='tls12'; iwr %nnabla_iwr_options% -Uri %1 -OutFile %FLATC_PACKAGE%.zip" || GOTO :error
    ECHO downloading %3
    MD %FLATC_PACKAGE%
    CD %FLATC_PACKAGE%
    cmake -E tar xvzf ..\%FLATC_PACKAGE%.zip || GOTO :error
    MOVE %2 %third_party_folder%\%3
    CD ..
    DEL %FLATC_PACKAGE%.zip
    RMDIR /s /q %FLATC_PACKAGE%
    exit /b

:error
ECHO failed with error code %errorlevel%.

exit /b %errorlevel%