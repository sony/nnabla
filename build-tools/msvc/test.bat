@ECHO OFF

REM Copyright 2018,2019,2020,2021 Sony Corporation.
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

SETLOCAL

REM Environment
CALL %~dp0tools\env.bat %1 || GOTO :error

REM Execute test

FOR /f %%i IN ('dir /b /s %nnabla_build_wheel_folder%\dist\nnabla-*.whl') DO set WHL=%%~fi
pip install %PIP_INS_OPTS% %WHL% || GOTO :error
pip install %PIP_INS_OPTS% pytest pytest-xdist[psutil]
python -m pytest %PYTEST_OPTS% %~dp0..\..\python\test || GOTO :error

CALL deactivate.bat  || GOTO :error

GOTO :end
:error
ECHO failed with error code %errorlevel%.
exit /b %errorlevel%

:end
ENDLOCAL

