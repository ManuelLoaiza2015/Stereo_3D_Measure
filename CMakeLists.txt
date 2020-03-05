cmake_minimum_required(VERSION 3.1)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set( CMAKE_BUILD_TYPE Release )
#set( CMAKE_BUILD_TYPE Debug )

if ( WIN32 )

	set( OpenCV_DIR D:/sdk/OpenCV/opencv/build_x32/install/)
	message( "WIN32: ${OpenCV_DIR}" )

else()

	set( OpenCV_DIR /home/manuel/Documents/sdk/OpenCV/installation/OpenCV-master/lib/cmake/opencv4 )
	message( "LINUX: ${OpenCV_DIR}" )

endif()

message( "build mode is: ${CMAKE_BUILD_TYPE}" )

project( Stereo_3D_Measure )

find_package( OpenCV REQUIRED )
find_package( Threads REQUIRED )

if ( WIN32 )
	message( "CONFIGURING FOR WINDOWS" )

else()
	message( "CONFIGURING FOR LINUX" )
	
endif()

add_subdirectory( src )