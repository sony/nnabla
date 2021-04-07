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

REM Build ZLIB

SET zlib_folder=%third_party_folder%\zlib123dllx64
SET zlib_library=%zlib_folder%\dll_x64\zlibwapi.lib
SET zlib_dll=%zlib_folder%\dll_x64\zlibwapi.dll
SET zlib_include_dir=%zlib_folder%
IF NOT EXIST %zlib_folder%.zip (
   powershell "iwr %nnabla_iwr_options% -Uri http://www.winimage.com/zLibDll/zlib123dllx64.zip -OutFile %zlib_folder%.zip" || GOTO :error
)

IF EXIST %zlib_library% (
   ECHO zlib already exists. Skipping...
   EXIT /b
)

CD %third_party_folder%

IF NOT EXIST %zlib_folder% (
	MD %zlib_folder%
)
CD %zlib_folder%
cmake -E tar xvzf ..\zlib123dllx64.zip || GOTO :error

IF NOT EXIST %third_party_folder%\zlib123.zip (
   powershell "[Net.ServicePointManager]::SecurityProtocol +='tls12'; iwr %nnabla_iwr_options% -Uri http://www.winimage.com/zLibDll/zlib123.zip -OutFile %third_party_folder%\zlib123.zip" || GOTO :error
)

CD %third_party_folder%
IF NOT EXIST zlib123 (
	MD zlib123
)
CD zlib123
cmake -E tar xvzf ..\zlib123.zip || GOTO :error
COPY zlib.h %zlib_folder% || GOTO :error
COPY zconf.h %zlib_folder% || GOTO :error
CD ..
cmake -E remove_directory zlib123

exit /b

:error
ECHO failed with error code %errorlevel%.

exit /b %errorlevel%
