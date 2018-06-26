function(download_and_extract_library NAME EXT URL)
  if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/${NAME})
    message("Downloading ${NAME}")
    file(DOWNLOAD
      ${URL}
      ${CMAKE_SOURCE_DIR}/third_party/${NAME}${EXT}
      SHOW_PROGRESS
      STATUS status
      LOG log)
    list(GET status 0 status_code)
    list(GET status 1 status_string)
    if(NOT status_code EQUAL 0)
      if(UNIX OR APPLE)
        message("Failed to download ${NAME}. Try to download with wget.")
        execute_process(
          COMMAND wget ${URL} -O ${CMAKE_SOURCE_DIR}/third_party/${NAME}${EXT})
      else()
        message(FATAL_ERROR "Failed to download ${NAME}. Please check network connection and proxy settings.
        status_code: ${status_code}
        status_string: ${status_string}
        log: ${log}
        ")
        error() # Place Unknown command to stop cmake process here.
      endif()
      
    endif()
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E
      chdir ${CMAKE_SOURCE_DIR}/third_party
      ${CMAKE_COMMAND} -E tar xf ${NAME}${EXT})
  endif()
endfunction()

download_and_extract_library(eigen-eigen-de14376a69f8 .zip https://bitbucket.org/eigen/eigen/get/de14376a69f8.zip)
download_and_extract_library(spdlog-0.16.3 .zip https://github.com/gabime/spdlog/archive/v0.16.3.zip)
if(BUILD_TEST)
  download_and_extract_library(googletest-release-1.7.0 .zip https://github.com/google/googletest/archive/release-1.7.0.zip)
  add_subdirectory(googletest-release-1.7.0)
endif()

if(BUILD_CPP_UTILS)
  download_and_extract_library(cmdline-master .zip https://github.com/tanakh/cmdline/archive/master.zip)
endif()
  
