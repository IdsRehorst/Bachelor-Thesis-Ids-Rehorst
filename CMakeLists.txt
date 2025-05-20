cmake_minimum_required(VERSION 3.20)
project(bachelor_thesis LANGUAGES C CXX)

#
# ---------------------------------------------------------------------------
# 1.  Global compiler settings
# ---------------------------------------------------------------------------
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#
# ---------------------------------------------------------------------------
# 2.  Build RACE *inside* this project with ExternalProject
# ---------------------------------------------------------------------------
#
include(ExternalProject)

# Where we will install RACE inside the build tree
set(RACE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/_deps/race-install")

ExternalProject_Add(race_external
        GIT_REPOSITORY  https://github.com/RRZE-HPC/RACE.git
        GIT_TAG         master
        SOURCE_DIR      ${CMAKE_BINARY_DIR}/_deps/race-src
        BINARY_DIR      ${CMAKE_BINARY_DIR}/_deps/race-build

        # ---- inject your custom CMakeLists.txt -------------------------
        PATCH_COMMAND
        ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/RACE_CMakeLists_IntelLLVM.txt"
        <SOURCE_DIR>/CMakeLists.txt
        COMMAND
        ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/matrixPower.cpp"
        <SOURCE_DIR>/src/matrixPower.cpp
        COMMAND
        ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/kernels.cpp"
        <SOURCE_DIR>/perf_test/src/helpers/kernels.cpp

        # ---- now configure/build with Intel oneAPI --------------------
        CMAKE_CACHE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=${RACE_INSTALL_PREFIX}
        -DCMAKE_C_COMPILER:FILEPATH=/opt/intel/oneapi/compiler/latest/bin/icx
        -DCMAKE_CXX_COMPILER:FILEPATH=/opt/intel/oneapi/compiler/latest/bin/icpx
        -DBUILD_SHARED_LIBS:BOOL=ON
        -DRACE_BUILD_TESTS:BOOL=OFF
        -DRACE_BUILD_EXAMPLES:BOOL=OFF
        -DRACE_BUILD_PERF_TESTS:BOOL=OFF
)

# This is where ExternalProject will install headers
file(MAKE_DIRECTORY "${RACE_INSTALL_PREFIX}/include")

set(RACE_LIB "${RACE_INSTALL_PREFIX}/lib/RACE/libRACE.so")

add_library(RACE::race SHARED IMPORTED GLOBAL)
set_target_properties(RACE::race PROPERTIES
        IMPORTED_LOCATION             "${RACE_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${RACE_INSTALL_PREFIX}/include")
add_dependencies(RACE::race race_external)

find_package(OpenMP REQUIRED)
find_package(MKL    REQUIRED)

#
# ---------------------------------------------------------------------------
# 3.  Library that belongs to your project
# ---------------------------------------------------------------------------
#
add_library(bachelor_thesis STATIC
        src/mmio.cpp
        src/sparsemat.cpp
        include/sparsemat.h
        src/coloring.cpp
        include/coloring.h
        src/solver.cpp
        include/solver.h)

target_link_libraries(bachelor_thesis PUBLIC
        RACE::race
        OpenMP::OpenMP_CXX
        MKL::MKL)

target_include_directories(bachelor_thesis PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

# Make sure we wait for the external build
add_dependencies(bachelor_thesis race_external)

#
# ---------------------------------------------------------------------------
# 4.  Executable
# ---------------------------------------------------------------------------
#
add_executable(tri_solve src/main.cpp)
target_link_libraries(tri_solve PRIVATE bachelor_thesis)

#
# ---------------------------------------------------------------------------
# 5. warnings / sanitizers
# ---------------------------------------------------------------------------
#
include(cmake/cpp_flags.cmake)   # your helper
# enable_warnings(bachelor_thesis tri_solve)
