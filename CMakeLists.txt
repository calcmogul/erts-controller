cmake_minimum_required(VERSION 3.21)

set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(erts_diff_drive)

include(FetchContent)

file(GLOB_RECURSE erts_diff_drive_src src/*.cpp)
add_executable(erts_diff_drive ${erts_diff_drive_src})

if(NOT MSVC)
    target_compile_options(erts_diff_drive PRIVATE -Wall -Wextra -pedantic)
endif()
set_property(TARGET erts_diff_drive PROPERTY COMPILE_WARNING_AS_ERROR ON)

target_compile_features(erts_diff_drive PUBLIC cxx_std_23)
target_include_directories(erts_diff_drive PRIVATE src)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.1
)
FetchContent_MakeAvailable(Eigen3)
target_link_libraries(erts_diff_drive PUBLIC Eigen3::Eigen)
