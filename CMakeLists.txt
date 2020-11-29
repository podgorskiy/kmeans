cmake_minimum_required(VERSION 3.6)

project(kmeans)

option (BUILD_EXAMPLE OFF)

add_library(kmeans INTERFACE)

target_include_directories(kmeans INTERFACE include/)

target_compile_features(kmeans INTERFACE cxx_std_11)


if(BUILD_EXAMPLE)
    option (USE_MSVC_RUNTIME_LIBRARY_DLL OFF)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)

    find_package( OpenCV REQUIRED )
    include_directories( ${OpenCV_INCLUDE_DIRS} )

    #####################################################################
    # Main sources
    #####################################################################
    file(GLOB SOURCES example/*.cpp)
    ############# ########################################################

    #####################################################################
    # Targets
    #####################################################################
    add_executable(kmeans_example ${SOURCES})
    #####################################################################

    ##############################################################
    # Setup compiler options and configurations
    ##############################################################
    if(MSVC)
        set(COMMON_OPTIONS /W3 /wd4996 /we4239 /we4002 /we4700 /we4305 /EHsc /MP)
        set(DEBUG_OPTIONS /MTd /Od /Zi)
        set(RELEASE_OPTIONS /MT /Ox /fp:fast /GL /GR-)

        target_compile_definitions(kmeans_example PRIVATE  -D_SCL_SECURE)
        target_compile_definitions(kmeans_example PRIVATE  -D_SECURE_SCL)
        target_compile_definitions(kmeans_example PRIVATE  -D_ALLOW_KEYWORD_MACROS)
        target_compile_definitions(kmeans_example PRIVATE  -D_CRT_SECURE_NO_WARNINGS)
        target_compile_definitions(kmeans_example PRIVATE  -D_CRT_SECURE_NO_DEPRECATE)
        target_compile_definitions(kmeans_example PRIVATE  -D_CRT_NONSTDC_NO_DEPRECATE)
        target_compile_definitions(kmeans_example PRIVATE  -DWINVER=0x0602)
        target_compile_definitions(kmeans_example PRIVATE  -D_WIN32_WINNT=0x0602)
        target_compile_definitions(kmeans_example PRIVATE  -DNTDDI_VERSION=0x06020000)

    elseif(APPLE)
        set(COMMON_OPTIONS -DHAVE_PTHREAD)
        set(DEBUG_OPTIONS -g)
        set (CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")
        set (CMAKE_C_FLAGS "-fPIC -g  ${CMAKE_C_FLAGS}")
        set(RELEASE_OPTIONS -Ofast -fno-strict-aliasing -msse2 -msse3 -msse4 -funsafe-math-optimizations)
    else()
        set(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")
        set(CMAKE_C_FLAGS "-fPIC -g  ${CMAKE_C_FLAGS}")
        set(COMMON_OPTIONS -DHAVE_PTHREAD)
        set(DEBUG_OPTIONS -g)
        set(RELEASE_OPTIONS -O2 -fPIC -fno-strict-aliasing -msse2 -msse3 -msse4 -funsafe-math-optimizations)
    endif()

    set(DEBUG_OPTIONS ${DEBUG_OPTIONS} ${COMMON_OPTIONS})
    set(RELEASE_OPTIONS ${RELEASE_OPTIONS} ${COMMON_OPTIONS})

    target_compile_options(kmeans_example PRIVATE
            "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>"
            "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")

    target_compile_definitions(kmeans_example PRIVATE  -D__STDC_LIMIT_MACROS)
    target_compile_definitions(kmeans_example PRIVATE  -D__STDC_FORMAT_MACROS)
    target_compile_definitions(kmeans_example PRIVATE  -D__STDC_CONSTANT_MACROS)

    target_compile_definitions(kmeans_example PRIVATE  -DHAVE_SSE42)


    if(MSVC)
    else()
        set(LIBRARIES rt m)
    endif()

    target_link_libraries(kmeans_example kmeans ${LIBRARIES} ${OpenCV_LIBS})
endif()
