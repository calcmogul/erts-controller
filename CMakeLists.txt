cmake_minimum_required(VERSION 3.21)

# Set default build type to release with debug info (i.e. release mode
# optimizations are performed, but debug info still exists).
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

project(ERTSDiffDrive)

# Generate compile_commands.json by default
if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS "YES" CACHE STRING "" FORCE)
endif()

file(GLOB_RECURSE ERTSDiffDrive_src src/*.cpp)
add_executable(ERTSDiffDrive ${ERTSDiffDrive_src})

target_compile_options(ERTSDiffDrive PRIVATE -Wall -Wextra -Werror)
target_compile_features(ERTSDiffDrive PUBLIC cxx_std_23)
target_include_directories(ERTSDiffDrive PRIVATE src)

include(FetchContent)

# Options for using a package manager (e.g., vcpkg) for certain dependencies
option(USE_SYSTEM_EIGEN "Use system eigen" OFF)

# Eigen dependency
if(NOT USE_SYSTEM_EIGEN)
    fetchcontent_declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        # master on 2024-06-14
        GIT_TAG 02bcf9b5918d46016bc88e5e9abebb6caa5a80b7
    )
    fetchcontent_makeavailable(Eigen3)
else()
    find_package(Eigen3 CONFIG REQUIRED)
endif()

target_link_libraries(ERTSDiffDrive PUBLIC Eigen3::Eigen)
