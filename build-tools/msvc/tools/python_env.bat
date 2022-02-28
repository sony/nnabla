REM Copyright 2020,2021 Sony Corporation.
REM Copyright 2021 Sony Group Corporation.
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
SET PYVER=%1
IF [%PYVER%] == [] (
   ECHO Please specify Python version 3.7, 3.8 or 3.9.
   EXIT /b 255
)
FOR /F "TOKENS=1 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MAJOR=%%A
FOR /F "TOKENS=2 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MINOR=%%A

REM VENV
IF [%BUILDID%] == [] SET BUILDID=local
SET VENV=%CD%\build-env-%BUILDID%-py%PYVER_MAJOR%%PYVER_MINOR%

if [%PYTHON_DIR%] == [] (
   SET PYTHON_DIR=C:\Python%PYVER_MAJOR%%PYVER_MINOR%
)

IF NOT EXIST "%PYTHON_DIR%" (
   ECHO "Please install python%PYVER_MAJOR%%PYVER_MINOR% with chocolatey".
   EXIT /b 255
)

%PYTHON_DIR%\python.exe -m venv %VENV%

IF NOT EXIST "%VENV%" (
   ECHO "Failed to create virtual env".
   EXIT /b 255
)

if [%VENV_PYTHON_PKG_DIR%] == [] (
   SET VENV_PYTHON_PKG_DIR=%VENV%\\Lib\\site-packages
)

CALL %VENV%\Scripts\activate.bat

CALL python -m pip install %PIP_INS_OPTS% --upgrade pip

CALL pip install %PIP_INS_OPTS% ^
           Cython ^
           boto3 ^
           h5py ^
           ipython ^
           librosa ^
           mako ^
           numpy ^
           pip ^
           protobuf ^
           pytest ^
           pywin32 ^
           pyyaml ^
           scikit-image ^
           scipy ^
           six ^
           tqdm ^
           virtualenv ^
           wheel
