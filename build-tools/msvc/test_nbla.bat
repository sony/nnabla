:: Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

%nnabla_build_folder%\bin\Release\test_nbla_utils || GOTO :error

ENDLOCAL

exit /b

:error
ECHO failed with error code %errorlevel%.
exit /b %errorlevel%
