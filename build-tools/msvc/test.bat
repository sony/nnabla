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

@ECHO ON

SETLOCAL

:: Settings
CALL %~dp0tools\default_settings.bat || GOTO :error

:: Folders
CALL %~dp0tools\default_folders.bat || GOTO :error

:: Execute test
IF EXIST %nnabla_test_venv_folder% RMDIR /s /q %nnabla_test_venv_folder%
python -m venv --system-site-packages %nnabla_test_venv_folder% || GOTO :error

CALL %nnabla_test_venv_folder%\scripts\activate.bat || GOTO :error

FOR /f %%i IN ('dir /b /s %nnabla_build_wheel_folder%\dist\*.whl') DO set WHL=%%~fi
pip install %WHL% || GOTO :error
python -m pytest %~dp0..\..\python\test || GOTO :error

CALL deactivate.bat  || GOTO :error

ENDLOCAL

exit /b

:error
ECHO failed with error code %errorlevel%.
exit /b %errorlevel%
