FROM intel/cpp-essentials:latest

# Force the OneAPI environment for all users and shells (non-interactive too)
RUN echo ". /opt/intel/oneapi/setvars.sh" >> /etc/profile.d/oneapi.sh
RUN apt update && apt install -y gdb

# Find Hwloc so RACE can build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libhwloc-dev pkg-config && \
    apt-get clean

RUN apt-get update && \
    apt-get install -y --no-install-recommends intel-oneapi-tbb-devel pkg-config

# Also set compilers explicitly for CMake
ENV CC=/opt/intel/oneapi/compiler/latest/bin/icx
ENV CXX=/opt/intel/oneapi/compiler/latest/bin/icpx

# Let CMake run from /workspace
WORKDIR /workspace

# Default to bash with environment sourced
CMD ["/bin/bash"]
