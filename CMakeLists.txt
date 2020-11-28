cmake_minimum_required(VERSION 3.6)

project(CV_HW1)

set(EXECUTABLE_NAME "CV_HW1")

if(MSVC)
    set(COMMON_OPTIONS /W3 /wd4996 /we4239 /we4002 /we4700 /we4305 /EHsc /MP)
    set(DEBUG_OPTIONS /MTd /Od /Zi)
    set(RELEASE_OPTIONS /MT /Ox /fp:fast /GL /GR-)
elseif(APPLE)
    set(COMMON_OPTIONS -DHAVE_PTHREAD)
    set(DEBUG_OPTIONS -g -Wall)
    set(RELEASE_OPTIONS -Ofast -fPIC -funsafe-math-optimizations -fno-strict-aliasing)
else()
    set(COMMON_OPTIONS -DHAVE_PTHREAD)
    set(DEBUG_OPTIONS -g -Wall)
    set(RELEASE_OPTIONS -Ofast -fPIC -funsafe-math-optimizations -fno-strict-aliasing)
endif()

set(DEBUG_OPTIONS ${DEBUG_OPTIONS} ${COMMON_OPTIONS})
set(RELEASE_OPTIONS ${RELEASE_OPTIONS} ${COMMON_OPTIONS})

add_compile_options(
        "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")

if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /DEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG /INCREMENTAL:NO")
    add_definitions(-DWINVER=0x0602)
    add_definitions(-D_WIN32_WINNT=0x0602)
    add_definitions(-DNTDDI_VERSION=0x06020000)
    add_definitions(-D__STDC_LIMIT_MACROS)
    add_definitions(-D__STDC_FORMAT_MACROS)
    add_definitions(-D__STDC_CONSTANT_MACROS)
    add_definitions(-D_HAS_EXCEPTIONS=0)
    add_definitions(-D_SCL_SECURE)
    add_definitions(-D_SECURE_SCL)
    add_definitions(-D_ALLOW_KEYWORD_MACROS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
    add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
endif()


if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set (CMAKE_CXX_FLAGS "--std=c++14 ${CMAKE_CXX_FLAGS}")
    endif ()
else ()
    set (CMAKE_CXX_STANDARD 14)
endif ()

include_directories(libs/stb)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Main sources
##############################################################
file(GLOB_RECURSE SOURCES sources/kmeans_segmentation.cpp sources/*.h)

add_executable(CV_HW3_kmeans ${SOURCES})

##############################################################
# Linkage
##############################################################
set(LIBRARIES rt m)

target_link_libraries(CV_HW3_kmeans ${LIBRARIES} ${OpenCV_LIBS})

