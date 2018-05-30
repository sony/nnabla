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

:: Build LibArchive

SET libarchive_tag=v3.3.2


SET libarchive_folder=%third_party_folder%\libarchive-%libarchive_tag%
SET libarchive_library=%libarchive_folder%\build-folder\libarchive\%build_type%\archive.lib
SET libarchive_dll=%libarchive_folder%\build-folder\bin\%build_type%\archive.dll
SET libarchive_include_dir=%libarchive_folder%\libarchive
IF NOT EXIST %libarchive_folder% (
	git clone https://github.com/libarchive/libarchive.git --branch %libarchive_tag% --depth=1 %libarchive_folder% || GOTO :error
)
CD %libarchive_folder%
IF NOT EXIST build-folder (
	MD build-folder
)
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
