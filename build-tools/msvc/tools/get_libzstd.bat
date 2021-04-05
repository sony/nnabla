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

REM Download pre-built zstd library

SET ZSTD_PACKAGE=zstd-v1.4.9
powershell "[Net.ServicePointManager]::SecurityProtocol +='tls12'; iwr %nnabla_iwr_options% -Uri https://github.com/facebook/zstd/releases/download/v1.4.9/zstd-v1.4.9-win64.zip -OutFile %ZSTD_PACKAGE%.zip" || GOTO :error

MD %ZSTD_PACKAGE%
CD %ZSTD_PACKAGE%
cmake -E tar xvzf ..\%ZSTD_PACKAGE%.zip || GOTO :error

MOVE dll\libzstd.dll %VENV%\Scripts\zstd.dll
CD ..

DEL %ZSTD_PACKAGE%.zip
RMDIR /s /q %ZSTD_PACKAGE%

:error
ECHO failed with error code %errorlevel%.

exit /b %errorlevel%
