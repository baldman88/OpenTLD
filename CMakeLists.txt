cmake_minimum_required(VERSION 2.8)

project(OpenTLD)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)

set(SOURCE_EXE main.cpp)

include_directories(opentld)

add_executable(OpenTLD ${SOURCE_EXE})

add_subdirectory(opentld)

target_link_libraries(OpenTLD  opentld
                               ${OpenCV_LIBS}
                               pthread)