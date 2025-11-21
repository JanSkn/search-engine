cmake_minimum_required(VERSION 3.15...4.0)

project(
  cpp_utils
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

find_library(STEMMER_LIB stemmer REQUIRED) 
find_path(STEMMER_INCLUDE_DIR NAMES libstemmer.h REQUIRED)

python_add_library(_core MODULE ./utils.cpp WITH_SOABI)

target_include_directories(_core PRIVATE
    ${STEMMER_INCLUDE_DIR}
)

target_link_libraries(_core PRIVATE
    pybind11::module             
    ${STEMMER_LIB}               
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cpp_utils
        DESTINATION .
        FILES_MATCHING PATTERN "*.py"
                       PATTERN "*.pyi")

target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _core DESTINATION cpp_utils)