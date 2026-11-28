# Hierarchical KV cache offloader

A C++ cache spills fixed-size KV blocks from a hot pool to host RAM and disk.

From the repo root, build and restore one spilled block with CMake 3.21+ and a C++20 compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/kv_offloader /tmp
```

[Storage, benchmarks, and tests](docs/reference.md)
