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

:: Build ZLIB

SET zlib_folder=%third_party_folder%\zlib123dllx64
SET zlib_library=%zlib_folder%\dll_x64\zlibwapi.lib
SET zlib_dll=%zlib_folder%\dll_x64\zlibwapi.dll
SET zlib_include_dir=%zlib_folder%
IF NOT EXIST %zlib_folder%.zip (
	powershell "iwr %nnabla_iwr_options% -Uri http://www.winimage.com/zLibDll/zlib123dllx64.zip -OutFile %zlib_folder%.zip" || GOTO :error
)
CD %third_party_folder%

IF NOT EXIST %zlib_folder% (
	MD %zlib_folder%
)
CD %zlib_folder%
cmake -E tar xvzf ..\zlib123dllx64.zip || GOTO :error

IF NOT EXIST %third_party_folder%\zlib123.zip (
   powershell "iwr %nnabla_iwr_options% -Uri http://www.winimage.com/zLibDll/zlib123.zip -OutFile %third_party_folder%\zlib123.zip" || GOTO :error
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
