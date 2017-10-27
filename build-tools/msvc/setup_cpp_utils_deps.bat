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

SETLOCAL
:: Options
SET protobuf_tag=v3.4.1
SET libarchive_tag=v3.3.2
SET build_type=Release
IF [%generate_target%] == [] (
  SET generate_target=Visual Studio 14 2015 Win64
)
SET nnabla_root=%~dp0..\..
SET third_party_folder=%nnabla_root%\third_party
IF [%nnabla_build_folder%] == [] (
  SET nnabla_build_folder=%nnabla_root%\build
)
SET nnabla_cmake_cpp_utils=%~dp0\cmake_cpp_utils.bat
SET wd=%cd%

:: Get ZLIB
SET zlib_folder=%third_party_folder%\zlib123dllx64
SET zlib_library=%zlib_folder%\dll_x64\zlibwapi.lib
SET zlib_dll=%zlib_folder%\dll_x64\zlibwapi.dll
SET zlib_include_dir=%zlib_folder%
if not exist %zlib_folder%.zip (
	powershell "iwr %nnabla_iwr_options% -Uri http://www.winimage.com/zLibDll/zlib123dllx64.zip -OutFile %zlib_folder%.zip" || GOTO :error
)
CD %third_party_folder%
SETLOCAL
if not exist %zlib_folder% (
	MD %zlib_folder%
)
CD %zlib_folder%
cmake -E tar xvzf ..\zlib123dllx64.zip || GOTO :error
ENDLOCAL

if not exist %third_party_folder%\zlib123.zip (
	powershell "iwr %nnabla_iwr_options% -Uri http://www.winimage.com/zLibDll/zlib123.zip -OutFile %third_party_folder%\zlib123.zip" || GOTO :error
)
SETLOCAL
if not exist zlib123 (
	MD zlib123
)
CD zlib123
cmake -E tar xvzf ..\zlib123.zip || GOTO :error
COPY zlib.h %zlib_folder% || GOTO :error
COPY zconf.h %zlib_folder% || GOTO :error
CD ..
cmake -E remove_directory zlib123
ENDLOCAL


:: Build LibArchive
SET libarchive_folder=%third_party_folder%\libarchive-%libarchive_tag%
SET libarchive_library=%libarchive_folder%\build-folder\libarchive\%build_type%\archive.lib
SET libarchive_dll=%libarchive_folder%\build-folder\bin\%build_type%\archive.dll
SET libarchive_include_dir=%libarchive_folder%\libarchive
if not exist %libarchive_folder% (
	git clone https://github.com/libarchive/libarchive.git --branch %libarchive_tag% --depth=1 %libarchive_folder% || GOTO :error
)
CD %libarchive_folder%
if not exist build-folder (
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
cmake.exe --build . --config Release || GOTO :error

:: Build protobuf libs
SET protobuf_folder=%third_party_folder%\protobuf-%protobuf_tag%
SET protobuf_include_dir=%protobuf_folder%\src
SET protobuf_bin_folder=%protobuf_folder%\build-folder\Release
SET protobuf_library=%protobuf_bin_folder%\libprotobuf.lib
SET protobuf_lite_library=%protobuf_bin_folder%\libprotobuf-lite.lib
SET protobuf_protoc_executable=%protobuf_bin_folder%\protoc.exe
if not exist %protobuf_folder% (
	git clone https://github.com/google/protobuf.git --branch %protobuf_tag% --depth=1 %protobuf_folder% || GOTO :error
)
CD %protobuf_folder%
if not exist build-folder (
	MD build-folder
)
CD build-folder
cmake.exe -G "%generate_target%" -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_TESTS=OFF ..\cmake || GOTO :error
cmake.exe --build . --config Release || GOTO :error
: extract_includes.bat

ECHO "--- CMake options for C++ build ---"
SET nnabla_cmake_command=cmake -G "%generate_target%" -DBUILD_CPP_UTILS=ON -DBUILD_PYTHON_PACKAGE=OFF -DLibArchive_LIBRARY=%libarchive_library% -DLibArchive_INCLUDE_DIR=%libarchive_include_dir% -DProtobuf_INCLUDE_DIR=%protobuf_include_dir% -DProtobuf_LIBRARY=%protobuf_library% -DProtobuf_LITE_LIBRARY=%protobuf_lite_library% -DProtobuf_PROTOC_EXECUTABLE=%protobuf_protoc_executable% -DPROTOC_COMMAND=%protobuf_protoc_executable% %nnabla_root%

@ECHO OFF
(
	ECHO SETLOCAL
	ECHO if not exist %nnabla_build_folder% ^(
	ECHO MD %nnabla_build_folder%
	ECHO ^)
	ECHO CD %nnabla_build_folder%
	ECHO %nnabla_cmake_command% ^|^| GOTO :error
	ECHO cmake --build . --config Release ^|^| GOTO :error
	ECHO :error
	ECHO ECHO failed with error code ^%errorlevel^%.
	ECHO exit /b ^%errorlevel^%
	ECHO ENDLOCAL
) > %nnabla_cmake_cpp_utils%
ENDLOCAL

:error
ECHO failed with error code %errorlevel%.
exit /b %errorlevel%

ENDLOCAL

