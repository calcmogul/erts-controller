cmake_minimum_required(VERSION 3.21)

# Set default build type to release with debug info (i.e. release mode
# optimizations are performed, but debug info still exists).
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

project(erts_diff_drive)

# Generate compile_commands.json by default
if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS "YES" CACHE STRING "" FORCE)
endif()

file(GLOB_RECURSE erts_diff_drive_src src/*.cpp)
add_executable(erts_diff_drive ${erts_diff_drive_src})

target_compile_options(erts_diff_drive PRIVATE -Wall -Wextra -Werror)
target_compile_features(erts_diff_drive PUBLIC cxx_std_23)
target_include_directories(erts_diff_drive PRIVATE src)

include(FetchContent)

# Options for using a package manager (e.g., vcpkg) for certain dependencies
option(USE_SYSTEM_EIGEN "Use system eigen" OFF)

# Eigen dependency
if(NOT USE_SYSTEM_EIGEN)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        # master on 2025-05-13
        GIT_TAG db85838ee2a2461e57a199cde4855c44f527fe07
    )
    FetchContent_MakeAvailable(Eigen3)
else()
    find_package(Eigen3 CONFIG REQUIRED)
endif()

target_link_libraries(erts_diff_drive PUBLIC Eigen3::Eigen)
