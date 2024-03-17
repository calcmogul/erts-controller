cmake_minimum_required(VERSION 3.21)

project(ERTSDiffDrive)

# Set default build type to release with debug info (i.e. release mode
# optimizations are performed, but debug info still exists).
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

# Generate compile_commands.json by default
if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS "YES" CACHE STRING "" FORCE)
endif()

file(GLOB_RECURSE ERTSDiffDrive_src src/*.cpp)
add_executable(ERTSDiffDrive ${ERTSDiffDrive_src})

target_compile_options(ERTSDiffDrive PRIVATE -Wall -Wextra -Werror)
target_compile_features(ERTSDiffDrive PUBLIC cxx_std_20)
target_include_directories(ERTSDiffDrive PRIVATE src)

include(FetchContent)

# Options for using a package manager (e.g., vcpkg) for certain dependencies
option(USE_SYSTEM_EIGEN "Use system eigen" OFF)
option(USE_SYSTEM_FMT "Use system fmt" OFF)

# Eigen dependency
if(NOT USE_SYSTEM_EIGEN)
    fetchcontent_declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        # master on 2024-03-06
        GIT_TAG 17f3bf8985021c242eedc9fa26c5c66d43a300f2
        PATCH_COMMAND
            git apply
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/eigen-disable-fortran-support.patch
        UPDATE_DISCONNECTED 1
    )
    fetchcontent_makeavailable(Eigen3)
else()
    find_package(Eigen3 CONFIG REQUIRED)
endif()

target_link_libraries(ERTSDiffDrive PUBLIC Eigen3::Eigen)

# fmt dependency
if(NOT USE_SYSTEM_FMT)
    fetchcontent_declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 10.2.1
    )
    fetchcontent_makeavailable(fmt)
else()
    find_package(fmt CONFIG REQUIRED)
endif()

target_link_libraries(ERTSDiffDrive PUBLIC fmt::fmt)
