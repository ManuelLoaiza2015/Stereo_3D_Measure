project( Camera_Capture )
message( "Configuring main application - Camera_Capture: " )

file(GLOB SOURCES_BASE  )

file(GLOB HEADERS_BASE )

file(GLOB SOURCES "*.cpp" )
file(GLOB HEADERS "*.h" )

include_directories( ${OpenCV_INCLUDE_DIRS} )

link_directories( )
 
add_executable( ${PROJECT_NAME}  ${HEADERS} 
${HEADERS_BASE} 
${SOURCES_BASE} 
${SOURCES} )
target_link_libraries( ${PROJECT_NAME} 
                       ${OpenCV_LIBS}
					   )	
                           