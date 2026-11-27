# Hierarchical KV cache offloader

A C++ cache spills fixed-size KV blocks from a hot pool to bounded host memory and then to a temporary file. A worker handles I/O and copies, and each prefetch returns a future. A resident lookup skips I/O. Its page lease pins the slot until the caller releases it. Keep the lease until the GPU work that uses the page completes.

## Run in host memory

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/kv_offloader /tmp
```

Both memory pools use DRAM. The demo inserts five blocks into two hot and two warm slots, then reads a resident block while the worker restores the oldest block from disk. It reports transfer counts, prefetch time, resident lookups during the restore, and a checksum over all restored bytes.

## Run with CUDA VRAM and NVMe

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build-cuda -j
./build-cuda/kv_offloader /mnt/nvme --gpu --direct
```

CUDA allocates the hot pool once and copies through a reusable pinned buffer on a dedicated nonblocking stream. For the spill file, Linux `--direct` requests `O_DIRECT` with aligned pages, and macOS requests `F_NOCACHE`. Use a directory on the NVMe volume with a filesystem that supports the requested mode. A failure in direct mode returns an error.

The demo uses resident lookups as a stand-in for model dispatch. A model scheduler must issue prefetches early enough to cover transfers and run another ready sequence while a required block is cold. A restore can fail if every hot slot has an active lease.

[CUDA runtime memory transfers](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)
