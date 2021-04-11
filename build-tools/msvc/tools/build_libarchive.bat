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

REM Build LibArchive

SET libarchive_folder=%third_party_folder%\libarchive-3.4.3
SET libarchive_library=%libarchive_folder%\build-folder\libarchive\%build_type%\archive.lib
SET libarchive_dll=%libarchive_folder%\build-folder\bin\%build_type%\archive.dll
SET libarchive_include_dir=%libarchive_folder%\libarchive


IF EXIST %libarchive_library% (
   IF EXIST %libarchive_dll% (
      ECHO libarchive already exists. Skipping...
      EXIT /b
   )
)

IF EXIST %libarchive_folder%.zip (
   DEL /Q %libarchive_folder%.zip
)


powershell "[Net.ServicePointManager]::SecurityProtocol +='tls12'; iwr %nnabla_iwr_options% -Uri https://github.com/libarchive/libarchive/archive/v3.4.3.zip -OutFile %libarchive_folder%.zip" || GOTO :error

RMDIR /S /Q %libarchive_folder%
CD %third_party_folder%
cmake -E tar xvzf %libarchive_folder%.zip || GOTO :error

CD %libarchive_folder%
MD build-folder

CD build-folder
cmake.exe -G "%generate_target%" .. ^
	-DENABLE_NETTLE=FALSE ^
	-DENABLE_OPENSSL=FALSE ^
	-DENABLE_LZO=FALSE ^
	-DENABLE_LZMA=FALSE ^
	-DENABLE_BZip2=FALSE ^
	-DENABLE_LIBXML2=FALSE ^
	-DENABLE_EXPAT=FALSE ^
	-DENABLE_PCREPOSIX=FALSE ^
	-DENABLE_LibGCC=FALSE ^
	-DENABLE_CNG=FALSE ^
	-DENABLE_TAR=FALSE ^
	-DENABLE_TAR_SHARED=FALSE ^
	-DENABLE_CPIO=FALSE ^
	-DENABLE_CPIO_SHARED=FALSE ^
	-DENABLE_CAT=FALSE ^
	-DENABLE_CAT_SHARED=FALSE ^
	-DENABLE_XATTR=FALSE ^
	-DENABLE_ACL=FALSE ^
	-DENABLE_ICONV=FALSE ^
	-DENABLE_TEST=FALSE ^
	-DZLIB_INCLUDE_DIR=%zlib_include_dir% ^
	-DZLIB_LIBRARY_RELEASE=%zlib_library% || GOTO :error
cmake.exe --build . --config %build_type% || GOTO :error

exit /b

:error
ECHO failed with error code %errorlevel%.

exit /b %errorlevel%
