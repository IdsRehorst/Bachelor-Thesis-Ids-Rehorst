cmake_minimum_required(VERSION 3.20)
project(bachelor_thesis LANGUAGES C CXX)


# ---------------------------------------------------------------------------
# 1.  Global compiler settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LIKWID REQUIRED CONFIG)
find_package(MKL REQUIRED CONFIG)
find_package(RACE REQUIRED CONFIG)
find_library(INTEL_OMP_LIB NAMES "iomp5")

message(STATUS "INTEL_OMP_LIB=${INTEL_OMP_LIB}")

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
		${RACE_INCLUDE_DIRS}
		${Kokkos_INCLUDE_DIRS}
                ${KokkosKernels_INCLUDE_DIRS}
                ${LIKWID_INCLUDES}
                )

target_link_libraries(bachelor_thesis PUBLIC
		${RACE_LIBRARIES}
		OpenMP::OpenMP_CXX
		MKL::MKL
		TBB::tbb
		${LIKWID_LIBRARIES}
		pthread
		Kokkos::kokkos
		Kokkos::kokkoskernels
		${OMP_LIBRARY})

target_compile_definitions(bachelor_thesis PUBLIC LIKWID_PERFMON)

# ---------------------------------------------------------------------------
# 4.  Executable
# ---------------------------------------------------------------------------
add_executable(tri_solve src/main.cpp)
target_link_libraries(tri_solve PRIVATE bachelor_thesis)

# ---------------------------------------------------------------------------
# 5.  Warnings / sanitizers (optional helper)
# ---------------------------------------------------------------------------
# enable_warnings(bachelor_thesis tri_solve)
