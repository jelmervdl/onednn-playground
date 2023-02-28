cmake_minimum_required(VERSION 3.10)
project("intel-moe")

# set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "Link to static DNNL library")
# set(DNNL_BUILD_TESTS OFF CACHE BOOL "Disable dnnl tests")
# set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "Disable dnnl examples")

# add_subdirectory("3rdparty/onednn")
# include_directories("3rdparty/onednn/include")

include(ExternalProject)
ExternalProject_Add(oneDNN
	PREFIX ${CMAKE_CURRENT_BINARY_DIR}/oneDNN
	UPDATE_COMMAND "" # Disable updating because we're using a fixed tag anyway
	GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
	GIT_TAG v3.0.1
	CMAKE_ARGS
		-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
		-DDNNL_LIBRARY_TYPE:STRING=STATIC
		-DDNNL_BUILD_TESTS:BOOL=OFF
		-DDNNL_BUILD_EXAMPLES:BOOL=OFF
	BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/oneDNN/lib/libdnnl.a
)

set(ONEDNN_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/oneDNN/include)
set(ONEDNN_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/oneDNN/lib)

include_directories(${ONEDNN_INCLUDE_DIR})

add_library(oneDNN::lib STATIC IMPORTED)
add_dependencies(oneDNN::lib oneDNN)
set_target_properties(oneDNN::lib PROPERTIES IMPORTED_LOCATION ${ONEDNN_LIB_DIR}/libdnnl.a)

add_subdirectory("src")

