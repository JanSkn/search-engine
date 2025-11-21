cmake_minimum_required(VERSION 3.20)
project(Indexer_Builder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ZLIB REQUIRED)

find_path(STEMMER_INCLUDE_DIR libstemmer.h)
find_library(STEMMER_LIBRARY stemmer)

set(SOURCES index_builder.cpp)

add_executable(index_builder ${SOURCES})

if (STEMMER_INCLUDE_DIR AND STEMMER_LIBRARY)
    target_include_directories(index_builder PRIVATE ${STEMMER_INCLUDE_DIR})
    target_link_libraries(index_builder PRIVATE ${STEMMER_LIBRARY})
else()
    message(WARNING "libstemmer not found: STEMMER_INCLUDE_DIR=${STEMMER_INCLUDE_DIR}, STEMMER_LIBRARY=${STEMMER_LIBRARY}")
endif()

target_link_libraries(index_builder PRIVATE ZLIB::ZLIB)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(index_builder PRIVATE stdc++fs)
endif()
