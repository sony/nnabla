# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################################
# Exclude target from all
#
function(nbla_exclude_from_all target_name)
  set_target_properties(
    ${target_name} PROPERTIES
    EXCLUDE_FROM_DEFAULT_BUILD TRUE
    EXCLUDE_FROM_ALL TRUE
    )
endfunction()


################################################################################################
# Find HDF5 source package
#
function(prepend lib_paths prefix)
  set(listvar "")
  foreach(f ${ARGN})
    if (CMAKE_BUILD_TYPE MATCHES Debug)
      list(APPEND listvar "${prefix}/lib${f}_debug.so")
    else()
      list(APPEND listvar "${prefix}/lib${f}.so")
    endif()
  endforeach(f)
  set (${lib_paths} "${listvar}" PARENT_SCOPE)
endfunction(prepend)

function(findhdf5)
  set(HDF5_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/third_party/hdf5-master/src
     ${PROJECT_SOURCE_DIR}/third_party/hdf5-master/hl/src
     ${CMAKE_BINARY_DIR}/third_party/hdf5-master)
  prepend(HDF5_LIBRARIES ${CMAKE_BINARY_DIR}/third_party/hdf5-master/bin/ ${HDF5_LIBRARIES_TO_EXPORT})
  set(HDF5_INCLUDE_DIRS "${HDF5_INCLUDE_DIRS}" PARENT_SCOPE)
  set(HDF5_LIBRARIES "${HDF5_LIBRARIES}" PARENT_SCOPE)
endfunction(findhdf5)



################################################################################################
# Clears variables from list
# Usage:
#   nbla_clear_vars(<variables_list>)
macro(nbla_clear_vars)
  foreach(_var ${ARGN})
    unset(${_var})
  endforeach()
endmacro()

################################################################################################
# Command for disabling warnings for different platforms (see below for gcc and VisualStudio)
# Usage:
#   nbla_warnings_disable(<CMAKE_[C|CXX]_FLAGS[_CONFIGURATION]> -Wshadow /wd4996 ..,)
macro(nbla_warnings_disable)
  set(_flag_vars "")
  set(_msvc_warnings "")
  set(_gxx_warnings "")

  foreach(arg ${ARGN})
    if(arg MATCHES "^CMAKE_")
      list(APPEND _flag_vars ${arg})
    elseif(arg MATCHES "^/wd")
      list(APPEND _msvc_warnings ${arg})
    elseif(arg MATCHES "^-W")
      list(APPEND _gxx_warnings ${arg})
    endif()
  endforeach()

  if(NOT _flag_vars)
    set(_flag_vars CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()

  if(MSVC AND _msvc_warnings)
    foreach(var ${_flag_vars})
      foreach(warning ${_msvc_warnings})
        set(${var} "${${var}} ${warning}")
      endforeach()
    endforeach()
  elseif((CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX) AND _gxx_warnings)
    foreach(var ${_flag_vars})
      foreach(warning ${_gxx_warnings})
        if(NOT warning MATCHES "^-Wno-")
          string(REPLACE "${warning}" "" ${var} "${${var}}")
          string(REPLACE "-W" "-Wno-" warning "${warning}")
        endif()
        set(${var} "${${var}} ${warning}")
      endforeach()
    endforeach()
  endif()
  nbla_clear_vars(_flag_vars _msvc_warnings _gxx_warnings)
endmacro()
