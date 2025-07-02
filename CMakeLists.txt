cmake_minimum_required(VERSION 3.20)
project(bachelor_thesis LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# RPATH for the GCC-11 runtime on DelftBlue
# ---------------------------------------------------------------------------
#set(GCC11_RT "/apps/generic/compiler/software/linux-rhel8-x86_64_v3/gcc-8.5.0/gcc-11.3.0-xl4qyfrehdkalgkfoku53ihtc7i4vf2z/lib64")
#set(CMAKE_INSTALL_RPATH "${GCC11_RT}")
#set(CMAKE_BUILD_RPATH   "${GCC11_RT}")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
#set(CMAKE_BUILD_WITH_INSTALL_RPATH  ON)

# ---------------------------------------------------------------------------
# 0.  LIKWID configuration  (marker API + runtime library)
# ---------------------------------------------------------------------------

# Either export LIKWID_ROOT before running cmake, or adapt the fallback path
set(LIKWID_ROOT $ENV{LIKWID_ROOT})
if(NOT LIKWID_ROOT)
	set(LIKWID_ROOT "$ENV{HOME}/likwid-5.4")
endif()

# Create imported target LIKWID::likwid
set(LIKWID_INCLUDE_DIR "${LIKWID_ROOT}/include")
set(LIKWID_LIB         "${LIKWID_ROOT}/lib/liblikwid.so")
if(NOT EXISTS "${LIKWID_LIB}")
	message(FATAL_ERROR "liblikwid.so not found in ${LIKWID_LIB}. "
			"Set LIKWID_ROOT env-var or edit CMakeLists.txt")
endif()

add_library(LIKWID::likwid SHARED IMPORTED)
set_target_properties(LIKWID::likwid PROPERTIES
		IMPORTED_LOCATION             "${LIKWID_LIB}"
		INTERFACE_INCLUDE_DIRECTORIES "${LIKWID_INCLUDE_DIR}")

set(LIKWID_HWLOC_LIB "${LIKWID_ROOT}/lib/liblikwid-hwloc.so")
add_library(LIKWID::likwid_hwloc SHARED IMPORTED)
set_target_properties(LIKWID::likwid_hwloc PROPERTIES IMPORTED_LOCATION "${LIKWID_HWLOC_LIB}")
# Add LIKW lib directory to the runtime search path
list(APPEND CMAKE_INSTALL_RPATH "${LIKWID_ROOT}/lib")
list(APPEND CMAKE_BUILD_RPATH   "${LIKWID_ROOT}/lib")
### LIKWID END ###############################################################

# ---------------------------------------------------------------------------
# 1.  Global compiler settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# 2.  Build RACE (ExternalProject)
# ---------------------------------------------------------------------------
include(ExternalProject)
set(RACE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/_deps/race-install")

ExternalProject_Add(race_external
		GIT_REPOSITORY  https://github.com/RRZE-HPC/RACE.git
		GIT_TAG         master
		SOURCE_DIR      ${CMAKE_BINARY_DIR}/_deps/race-src
		BINARY_DIR      ${CMAKE_BINARY_DIR}/_deps/race-build
		# inject patched files
		PATCH_COMMAND
		${CMAKE_COMMAND} -E copy
		"${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/RACE_CMakeLists_IntelLLVM.txt"
		<SOURCE_DIR>/CMakeLists.txt
		COMMAND ${CMAKE_COMMAND} -E copy
		"${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/matrixPower.cpp"
		<SOURCE_DIR>/src/matrixPower.cpp
		COMMAND ${CMAKE_COMMAND} -E copy
		"${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/kernels.cpp"
		<SOURCE_DIR>/perf_test/src/helpers/kernels.cpp
		COMMAND
	        ${CMAKE_COMMAND} -E copy
		"${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/functionManager.cpp"
	        <SOURCE_DIR>/src/functionManager.cpp
	   	COMMAND
       		${CMAKE_COMMAND} -E copy
       		"${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/pin.cpp"
       		<SOURCE_DIR>/src/pin.cpp

		# configure & build
		CMAKE_CACHE_ARGS
		-DCMAKE_INSTALL_PREFIX:PATH=${RACE_INSTALL_PREFIX}
		-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
		-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
		-DBUILD_SHARED_LIBS:BOOL=ON
		-DRACE_BUILD_TESTS:BOOL=OFF
		-DRACE_BUILD_EXAMPLES:BOOL=OFF
		-DRACE_BUILD_PERF_TESTS:BOOL=OFF
		-DHWLOC_INCLUDE_DIR:PATH=$ENV{HWLOC_ROOT}/include
		-DLIBHWLOC:FILEPATH=$ENV{HWLOC_ROOT}/lib/libhwloc.so
		-DMKL_THREADING:STRING=gnu_thread)

file(MAKE_DIRECTORY "${RACE_INSTALL_PREFIX}/include")

add_library(RACE::race SHARED IMPORTED GLOBAL)
set_target_properties(RACE::race PROPERTIES
		IMPORTED_LOCATION             "${RACE_INSTALL_PREFIX}/lib/RACE/libRACE.so"
		INTERFACE_INCLUDE_DIRECTORIES "${RACE_INSTALL_PREFIX}/include")
add_dependencies(RACE::race race_external)

find_package(OpenMP REQUIRED)
set(MKL_INTERFACE "lp64")
find_package(MKL REQUIRED)

set(TBB_ROOT $ENV{TBBROOT})
find_package(TBB CONFIG REQUIRED)

find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

get_target_property(KKK_INCS Kokkos::kokkoskernels INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS ">>> Kokkos::kokkoskernels INTERFACE_INCLUDE_DIRECTORIES = ${KKK_INCS}")

# ---------------------------------------------------------------------------
# 3.  Project library
# ---------------------------------------------------------------------------
add_library(bachelor_thesis STATIC
		src/mmio.cpp
		src/sparsemat.cpp   include/sparsemat.h
		src/coloring.cpp    include/coloring.h
		src/solver.cpp      include/solver.h)

target_include_directories(bachelor_thesis PUBLIC
		$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
		${Trilinos_INCLUDE_DIRS}
                ${Trilinos_Tpetra_INCLUDE_DIRS})

target_link_libraries(bachelor_thesis PUBLIC
		RACE::race
		OpenMP::OpenMP_CXX
		MKL::MKL
		TBB::tbb
		LIKWID::likwid
		LIKWID::likwid_hwloc
		pthread
		Kokkos::kokkos
		Kokkos::kokkoskernels)     

target_compile_definitions(bachelor_thesis PUBLIC LIKWID_PERFMON)

add_dependencies(bachelor_thesis race_external)

# ---------------------------------------------------------------------------
# 4.  Executable
# ---------------------------------------------------------------------------
add_executable(tri_solve src/main.cpp)
target_link_libraries(tri_solve PRIVATE bachelor_thesis)

set_target_properties(tri_solve PROPERTIES
		BUILD_WITH_INSTALL_RPATH  ON
		INSTALL_RPATH             "$ORIGIN/_deps/race-install/lib/RACE;${GCC11_RT};${LIKWID_ROOT}/lib"
		SKIP_BUILD_RPATH          OFF)

# ---------------------------------------------------------------------------
# 5.  Warnings / sanitizers (optional helper)
# ---------------------------------------------------------------------------
include(cmake/cpp_flags.cmake)
# enable_warnings(bachelor_thesis tri_solve)
