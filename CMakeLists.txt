cmake_minimum_required(VERSION 3.21)
project(kv_offloader LANGUAGES CXX)

option(ENABLE_CUDA "Use a CUDA hot tier" OFF)
option(KV_TEST_DIRECT_IO "Exercise direct I/O on filesystems that support it" OFF)
find_package(Threads REQUIRED)
add_library(kv_cache STATIC cache.cpp storage.cpp)
target_include_directories(kv_cache PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(kv_cache PUBLIC cxx_std_20)
target_link_libraries(kv_cache PRIVATE Threads::Threads)
add_executable(kv_offloader main.cpp)
target_link_libraries(kv_offloader PRIVATE kv_cache)

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(kv_cache PUBLIC OFFLOADER_CUDA)
    target_link_libraries(kv_cache PRIVATE CUDA::cudart)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(kv_cache_tests tests/cache_tests.cpp)
    target_link_libraries(kv_cache_tests PRIVATE kv_cache)
    add_test(NAME kv_cache_lifecycle COMMAND kv_cache_tests)
    add_test(NAME kv_cache_cli
        COMMAND "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:kv_offloader>"
                "-DSTATE_DIR=${CMAKE_CURRENT_BINARY_DIR}" "-DCUDA_ENABLED=${ENABLE_CUDA}"
                "-DDIRECT_IO=${KV_TEST_DIRECT_IO}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli.cmake")
    set_tests_properties(kv_cache_lifecycle kv_cache_cli PROPERTIES TIMEOUT 30)
endif()
