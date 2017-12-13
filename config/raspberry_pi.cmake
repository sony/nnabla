# Make sure cmake do not compile test program
INCLUDE(CMakeForceCompiler)
CMAKE_FORCE_CXX_COMPILER(arm-linux-gnueabihf-g++ GNU)
CMAKE_FORCE_C_COMPILER(arm-linux-gnueabihf-gcc GNU)

# This must be set after CMAKE_FORCE_CXX_COMPILER
SET(CMAKE_CXX_COMPILER ${NBLA_TOOLCHAIN_ROOT}/bin/arm-linux-gnueabihf-g++)
SET(CMAKE_C_COMPILER ${NBLA_TOOLCHAIN_ROOT}/bin/arm-linux-gnueabihf-gcc)
SET(CMAKE_FIND_ROOT_PATH /)
SET(CMAKE_SYSROOT ${NBLA_SYSROOT})
list(APPEND NBLA_INCLUDE_DIRS
	${NBLA_PROTOBUF}/src)

SET(LibArchive_LIBRARY ${NBLA_SYSROOT}/usr/lib/arm-linux-gnueabihf/libarchive.so.13)
SET(ZLIB_LIBRARY ${NBLA_SYSROOT}/usr/lib/arm-linux-gnueabihf/libz.so)
SET(PROTOBUF_LIBRARY ${NBLA_SYSROOT}/usr/lib/arm-linux-gnueabihf/libprotobuf.so)
