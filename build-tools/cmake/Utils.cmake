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
