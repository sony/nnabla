@ECHO OFF

REM Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

REM Download pre-built lz4 library

SET LZ4_PACKAGE=lz4-v1.9.3
powershell "[Net.ServicePointManager]::SecurityProtocol +='tls12'; iwr %nnabla_iwr_options% -Uri https://github.com/lz4/lz4/releases/download/v1.9.3/lz4_win64_v1_9_3.zip -OutFile %LZ4_PACKAGE%.zip" || GOTO :error

MD %LZ4_PACKAGE%
CD %LZ4_PACKAGE%
cmake -E tar xvzf ..\%LZ4_PACKAGE%.zip || GOTO :error

MOVE dll\msys-lz4-1.dll %VENV%\Scripts\liblz4.dll
CD ..

DEL %LZ4_PACKAGE%.zip
RMDIR /s /q %LZ4_PACKAGE%

:error
ECHO failed with error code %errorlevel%.

exit /b %errorlevel%
