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

REM Build hdf5 for windows

SET hdf5_folder=%third_party_folder%\hdf5-master

CD %third_party_folder%

call :download_hdf5_and_rename  https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_12_2.zip

CD %third_party_folder%\hdf5-master
IF NOT EXIST build-folder (
   MD build-folder
)
CD %third_party_folder%\hdf5-master\build-folder

cmake.exe -G "%generate_target%" .. ^
          -DBUILD_STATIC_LIBS=OFF ^
          -DONLY_SHARED_LIBS=ON ^
          -DBUILD_SHARED_LIBS=ON ^
          -DCPACK_SOURCE_ZIP=OFF ^
          -DHDF5_BUILD_HL_TOOLS=OFF ^
          -DHDF5_BUILD_TOOLS=OFF ^
          -DHDF5_BUILD_UTILS=OFF ^
          -DHDF5_BUILD_EXAMPLES=OFF ^
          -DHDF5_TEST_CPP=OFF ^
          -DHDF5_TEST_EXAMPLES=OFF ^
          -DHDF5_TEST_JAVA=OFF ^
          -DHDF5_TEST_TOOLS=OFF ^
          -DHDF5_TEST_VFD=OFF ^
          -DHDF5_TEST_SWMR=OFF ^
          -DHDF5_TEST_PARALLEL=OFF ^
          -DHDF5_BUILD_HL_LIB=ON || GOTO :error
cmake.exe --build . --config %build_type% || GOTO :error

exit /b


:download_hdf5_and_rename
    IF NOT EXIST hdf5-master (
        powershell "[Net.ServicePointManager]::SecurityProtocol +='tls12'; iwr %nnabla_iwr_options% -Uri %1 -OutFile hdf5-master.zip" || GOTO :error
        ECHO downloading %1
        cmake -E tar xvzf hdf5-master.zip || GOTO :error
        MOVE hdf5-hdf5-1_12_2 hdf5-master
    )
    exit /b


:error
ECHO failed with error code %errorlevel%.
exit /b %errorlevel%

