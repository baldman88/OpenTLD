cmake_minimum_required(VERSION 2.8)

#set(CMAKE_C_FLAGS "-O3 -Wall")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(OpenTLD)

find_package(OpenCV REQUIRED)

set(SOURCES 	main.cpp
		opentld/Classifier.cpp
		opentld/Detector.cpp
		opentld/Feature.cpp
		opentld/Fern.cpp
		opentld/KalmanFilter.cpp
		opentld/Patch.cpp
		opentld/TLDTracker.cpp
		opentld/Tracker.cpp
		opentld/Leaf.cpp)

add_executable(OpenTLD ${SOURCES})

target_link_libraries(OpenTLD  ${OpenCV_LIBS}
                               pthread)