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

FOR /F "TOKENS=1 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MAJOR=%%A
FOR /F "TOKENS=2 DELIMS=." %%A IN ("%PYVER%") DO SET PYVER_MINOR=%%A

IF [%BUILDID%] == [] (
   ECHO BUILDID is not set
   EXIT /b
)

SET VENV=%CD%\build-env-%BUILDID%-py%PYVER_MAJOR%%PYVER_MINOR%

IF [%BUILDID%] == [local] (
   ECHO Skip Cleanup local virtual environment.
   EXIT /b
)

ECHO Cleanup virtual environment.
CALL %VENV%\Scripts\activate.bat
CALL %VENV%\Scripts\deactivate.bat

RMDIR /s /q %VENV%

REM Ignore error.
EXIT /b 0
