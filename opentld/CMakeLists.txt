cmake_minimum_required(VERSION 2.8)

project(opentld)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)

set(SOURCES 	Classifier.cpp
		Detector.cpp
		Feature.cpp
		Fern.cpp
		Leaf.cpp
		KalmanFilter.cpp
		Patch.cpp
		TLDTracker.cpp
		Tracker.cpp)

#aux_source_directory("." SOURCES)

add_library(opentld STATIC ${SOURCES})