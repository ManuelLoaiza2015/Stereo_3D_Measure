project( Stereo_System_Application )
message( "Configuring main application - Stereo_System_Application: " )

file(GLOB SOURCES_BASE  )
file(GLOB HEADERS_BASE )

file(GLOB SOURCES "*.cpp" )
file(GLOB HEADERS "*.h" )

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

#add_executable(test test.cpp)
#target_link_libraries(test Threads::Threads)

include_directories( ${OpenCV_INCLUDE_DIRS} )

link_directories( )
 
add_executable( ${PROJECT_NAME}  ${HEADERS} 
${HEADERS_BASE} 
${SOURCES_BASE} 
${SOURCES} )
target_link_libraries( ${PROJECT_NAME} 
                       ${OpenCV_LIBS} 
                       Threads::Threads )	
                           