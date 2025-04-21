cmake_minimum_required(VERSION 3.20)
project(bachelor_thesis LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# 1.  Compiler settings that apply to *every* target created afterwards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # lets IDEs find headers

# ---------------------------------------------------------------------------
# 2.  Dependencies – do this **before** you declare targets
# ---------------------------------------------------------------------------
find_package(RACE REQUIRED CONFIG)     
find_package(OpenMP REQUIRED)

# ---- Library -----------------------------------------------------------
add_library(bachelor_thesis STATIC src/colorizer.cpp)

# 1.  Add the RACE headers
target_include_directories(bachelor_thesis
    PUBLIC
        ${RACE_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

# 2.  Link to the actual libraries RACE told us about
target_link_libraries(bachelor_thesis
    PUBLIC
        ${RACE_LIBRARIES}          
        OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# 4.  Application target
# ---------------------------------------------------------------------------
add_executable(tri_solve src/main.cpp)
target_link_libraries(tri_solve PRIVATE bachelor_thesis)

# ---------------------------------------------------------------------------
# 5.  Compiler warnings (after targets exist!)
# ---------------------------------------------------------------------------
include(cmake/cpp_flags.cmake)      # now it can “see” the targets
enable_warnings(bachelor_thesis tri_solve)   # function defined in that file

