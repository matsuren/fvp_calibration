cmake_minimum_required(VERSION 3.1)

include_directories(.)
aux_source_directory(common COMMON_SRC)
set(APRILTAG_SRCS apriltag.c apriltag_pose.c apriltag_quad_thresh.c)

# Library
set(TAG_FILES tag36h11.c tagCircle21h7.c)
# file(GLOB TAG_FILES ${CMAKE_SOURCECMAKE_CURRENT_SOURCE_DIR_DIR}/tag*.c)
set_source_files_properties(SOURCE ${TAG_FILES} PROPERTIES COMPILE_FLAGS -O0)
add_library(apriltag STATIC ${APRILTAG_SRCS} ${COMMON_SRC} ${TAG_FILES})
if (MSVC)
    # FindThreads will not find pthread.h with MSVC
    # winmm is necessary for __imp_timeGetTime
    target_link_libraries(apriltag winmm)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(apriltag Threads::Threads m)
endif()

set_target_properties(apriltag PROPERTIES SOVERSION 3 VERSION 3.1.0)

include(GNUInstallDirs)
target_include_directories(apriltag PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>/apriltag")

set_target_properties(apriltag PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_INSTALL_LIBDIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_LIBDIR}
)
