cmake_minimum_required(VERSION 3.21)
project(kv_offloader LANGUAGES CXX)

option(ENABLE_CUDA "Use a CUDA hot tier" OFF)
find_package(Threads REQUIRED)
add_executable(kv_offloader main.cpp cache.cpp storage.cpp)
target_compile_features(kv_offloader PRIVATE cxx_std_20)
target_link_libraries(kv_offloader PRIVATE Threads::Threads)

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(kv_offloader PRIVATE OFFLOADER_CUDA)
    target_link_libraries(kv_offloader PRIVATE CUDA::cudart)
endif()
