# Hierarchical KV cache offloader reference

Run the commands from the repository root.

This C++ cache spills fixed-size KV blocks from a hot pool to bounded host RAM and then to a temporary file. A FIFO worker handles transfers and reclamation.

## Run the CPU demo

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/kv_offloader /tmp
./build/kv_offloader /tmp --disk-slots 4 --churn 256
```

The short demo restores and verifies one spilled block. The churn run fills two hot slots, two host slots, and three disk slots with 64-KiB pages. It checks disk pressure, lease validity after retirement, and retirement queued behind a restore. It then runs 256 cycles of restore, checksum, retire, and insert.

### Recorded CPU churn

The buffered run on an Apple M3 Pro kept seven live blocks, reused 256 spill slots, and reached the 262,144-byte file budget. It reported zero integrity failures. The 256-cycle loop took 98.7 ms, including fixture generation, checksums, and cache operations.

## Page lifetime

- `try_get` only looks up resident blocks. `prefetch` returns a future for the transfer. A page lease pins the slot until its last holder releases it. Hold the lease until device work using the page completes.
- `retire(id)` takes effect in worker submission order. Earlier prefetches may complete first. Once retirement completes, lookups for the ID miss. Existing hot leases still read the same bytes, and outstanding leases can prevent eviction even after retirement.
- Use a distinct ID for each block generation if an older operation may still be in flight.

### Capacity and promotion

The spill file reuses released slots and is bounded by `disk_slots × page_bytes`. Normal spills leave one scratch slot free so a full cache can restore a disk page without overwriting it first. The hot pool reserves one extra physical page beyond its resident capacity to stage promotion before eviction.

The worker copies the hot victim to host memory before using the disk scratch slot. It keeps the original disk page until it can publish the replacement. A failed copy or slot allocation leaves the old page available and the scratch slot reusable.

When the normal spill budget is full, an insert that needs another spill returns pressure. File usage tracks the highest successfully written offset, so reclaiming a slot can leave file size unchanged.

### Spill integrity

The worker records an in-memory FNV-1a checksum after a full disk write. Prefetch verifies it before eviction or publication. A short read or checksum mismatch fails the future and increments `disk_integrity_failures`. Read system-call errors propagate to the caller. Retire a corrupt block and insert rebuilt KV bytes to recover. The checksum detects accidental corruption and adds one pass per page on write and on read.

## Run with CUDA VRAM and NVMe

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build-cuda -j
./build-cuda/kv_offloader /mnt/nvme --gpu --direct --churn 256
```

CUDA allocates the hot pool and uses one pinned staging buffer on a dedicated stream. The worker synchronizes each copy before reusing the buffer. With `--direct`, Linux requests `O_DIRECT` and macOS requests `F_NOCACHE`. Use a filesystem that supports the selected mode.

The CUDA/NVMe command has no recorded run. The demo polls resident blocks through cache lookups. Model dispatch and decode latency are outside those measurements. Serving requires a scheduler that issues prefetches early and switches to ready sequences while cold pages load.

[CUDA runtime memory transfers](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda-runtime-api/group__CUDART__MEMORY.html)

## Tests and coverage

The tests cover temporary-file spills, bounded slot reuse, queued retirement, old lease lifetime, and foreign-lease rejection. Fault tests truncate a page, force a read error, and flip one byte of a full page. They verify that a rejected promotion leaves the old hot page and disk scratch slot available, and that retirement allows a corrupt slot to hold valid data again.

Hot and spill transfers require a host buffer of exactly one configured page. The size check runs before copying bytes, changing slot state, or updating corruption counters. Hot storage rejects duplicate releases, inactive slots, out-of-range indices, and overflowing allocation sizes. Keep a slot claimed while accessing its bytes.

### Coverage command

Direct I/O tests are opt-in. This command enables them.

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

On Linux, use `llvm-profdata` and `llvm-cov` from the same Clang toolchain without the `xcrun` prefix.

The recorded AppleClang CPU build with direct I/O enabled covered 599/619 lines (96.77%) and 166/200 branches (83.00%). It covered all 142 instrumented lines in `storage.cpp`. CUDA verification requires a separate build and an NVIDIA device.
