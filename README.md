# Hierarchical KV cache offloader

This C++ cache spills fixed-size KV blocks from a hot pool to bounded host RAM and then to a temporary file. A FIFO worker handles transfers and reclamation. `try_get` only looks up resident blocks. `prefetch` returns a future for the transfer, and the resulting page lease pins its slot until the caller releases it. Hold the lease until device work using the page has completed.

## Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/kv_offloader /tmp
./build/kv_offloader /tmp --disk-slots 4 --churn 256
```

The short demo restores and verifies one spilled block. The churn run fills two hot slots, two host slots, and three disk slots with 64-KiB pages. It exercises disk pressure and verifies an existing lease after retirement and retirement submitted behind a queued restore. It then cycles through 256 restore, checksum, retire, and insert operations.

On an Apple M3 Pro, the buffered run kept seven live blocks, reused 256 spill slots, and reached the 262,144-byte file budget. It reported zero integrity failures. The 256-cycle loop took 98.7 ms in this run, including fixture generation, checksums, and cache operations.

## Retirement and disk slots

`retire(id)` takes effect in worker submission order. After it completes, the ID leaves the cache and subsequent lookups miss. Earlier prefetches may complete before retirement. A pre-existing hot lease still reads the same bytes and keeps its slot allocated until the final holder releases it. Use a distinct ID for each block generation if an older operation may still be in flight.

The spill file reuses released slots and is bounded by `disk_slots × page_bytes`. The normal spill path keeps one slot available so that a full cache can restore a disk page without first overwriting it. The hot allocator reserves one extra physical page beyond its resident capacity to stage promotion before eviction. The worker copies the hot victim to host memory before consuming the disk scratch slot, and keeps the original disk page until it can publish the replacement. A failed copy or slot allocation therefore leaves the old page available and the disk scratch slot reusable. A new insert that needs another spill returns pressure once the normal budget is full. Outstanding leases can prevent eviction even after a block has been retired. File usage counts the highest successfully written offset, so reclaiming a slot can leave the file size unchanged.

The worker records an in-memory FNV-1a checksum after a full disk write. A prefetch checks the bytes before eviction or publication. A mismatch fails the future and increments `disk_integrity_failures`. The caller can retire the corrupt block and insert rebuilt KV bytes. The checksum adds one pass over each page on write and read and detects accidental corruption.

## Run with CUDA VRAM and NVMe

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build-cuda -j
./build-cuda/kv_offloader /mnt/nvme --gpu --direct --churn 256
```

CUDA allocates the hot pool and uses one pinned staging buffer on a dedicated stream. The worker synchronizes each copy before reusing the buffer. Linux `--direct` requests `O_DIRECT`, and macOS requests `F_NOCACHE`. Use a filesystem supporting the selected mode.

The CUDA/NVMe command has not been run. Real serving requires a scheduler that issues prefetches early and switches to ready sequences while cold pages load. The demo's resident polls are cache lookups; there is no model dispatch or measurement of decode latency.

[CUDA runtime memory transfers](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda-runtime-api/group__CUDART__MEMORY.html)

## Tests and coverage

Tests exercise temporary-file spills, bounded slot reuse, queued retirement, old lease lifetime, and foreign-lease rejection. They truncate a page, force a read error, and flip one byte of a full page. A rejected promotion leaves the old hot page and the disk scratch slot available. The corruption case also verifies that retirement lets the slot hold valid data again. Direct I/O tests are opt-in.

```sh
cmake -S . -B build/coverage -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" -DKV_TEST_DIRECT_IO=ON
cmake --build build/coverage -j
profiles="$(mktemp -d)"
LLVM_PROFILE_FILE="$profiles/%m-%p.profraw" ctest --test-dir build/coverage --output-on-failure
xcrun llvm-profdata merge -sparse "$profiles"/*.profraw -o "$profiles/merged.profdata"
xcrun llvm-cov report build/coverage/kv_offloader \
  -object build/coverage/kv_cache_tests -instr-profile="$profiles/merged.profdata" \
  --show-branch-summary main.cpp cache.cpp cache.hpp storage.cpp storage.hpp
```

On Linux, use `llvm-profdata` and `llvm-cov` from the same Clang toolchain without the `xcrun` prefix. The instrumented AppleClang CPU build with direct I/O enabled covered 599/619 lines (96.77%) and 166/200 branches (83.00%). It covered all 142 instrumented lines in `storage.cpp`. CUDA verification requires a separate build and an NVIDIA device.
