cmake_minimum_required(VERSION 3.12...3.30)

# Set the project name
project(c-mnist-nn)

# Try to find a locally installed raylib, but don't quit on fail
find_package(raylib 5.0 QUIET)

# This code downloads raylib into a directory called _deps and adds it as a subdirectory
include(FetchContent)
if (NOT raylib_FOUND)
    FetchContent_Declare(
        raylib
        URL https://github.com/raysan5/raylib/archive/refs/tags/5.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP True
    )
    FetchContent_MakeAvailable(raylib)
endif()

# We don't want raylib's examples built
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Collect all source files
set(SOURCES
    src/main.c
    src/gui.c
    src/nn.c
    src/train.c
    src/viz.c
)

# Here, the executable is declared with all its sources
add_executable(main ${SOURCES})

# Link raylib to main
target_link_libraries(main 
    raylib
)

# Make main find the header files
target_include_directories(main 
    PUBLIC "${raylib_SOURCE_DIR}/src"
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src"  # Add this line to find your header files
)

if (EMSCRIPTEN)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lidbfs.js -s USE_GLFW=3 --shell-file ${CMAKE_CURRENT_LIST_DIR}/web/minshell.html --preload-file ${CMAKE_CURRENT_LIST_DIR}/resources/@resources/ -s GL_ENABLE_GET_PROC_ADDRESS=1")
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
endif ()