cmake_minimum_required(VERSION 3.9)

set(ROUTINE spmm_blockedell)

project("${ROUTINE}_example"
        DESCRIPTION  "GPU-Accelerated Sparse Linear Algebra"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cusparse/index.html"
        LANGUAGES    CXX)

add_executable(${ROUTINE}_example)

target_sources(${ROUTINE}_example
    PUBLIC ${PROJECT_SOURCE_DIR}/${ROUTINE}_example.cpp
)

target_include_directories(${ROUTINE}_example
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${ROUTINE}_example
    PUBLIC cudart cusparse
)
