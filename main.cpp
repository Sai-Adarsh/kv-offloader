#include "cache.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t page_bytes = 64 * 1024;
constexpr KvCache::BlockId oldest_id = 0;
constexpr KvCache::BlockId active_id = 4;

std::vector<std::byte> fixture(KvCache::BlockId id) {
    std::vector<std::byte> result(page_bytes);
    for (std::size_t i = 0; i < page_bytes; ++i) {
        result[i] = static_cast<std::byte>((id * 31 + i + i / 11) & 0xff);
    }
    return result;
}

std::uint64_t checksum(const std::vector<std::byte>& payload) {
    std::uint64_t value = 14695981039346656037ull;
    for (const std::byte byte : payload) {
        value = (value ^ std::to_integer<unsigned char>(byte)) * 1099511628211ull;
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        throw std::invalid_argument("usage: kv_offloader SPILL_DIRECTORY [--gpu] [--direct]");
    }
    bool gpu = false;
    bool direct = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--gpu") {
            gpu = true;
        } else if (flag == "--direct") {
            direct = true;
        } else {
            throw std::invalid_argument("unknown kv_offloader flag");
        }
    }

    KvCache cache(2, 2, page_bytes, std::filesystem::path(argv[1]), gpu, direct);
    for (KvCache::BlockId id = oldest_id; id <= active_id; ++id) {
        cache.insert(id, fixture(id)).get();
    }
    if (cache.try_get(oldest_id)) {
        throw std::runtime_error("expected at least one block on disk");
    }
    const auto pin = cache.try_get(active_id);
    if (!pin) {
        throw std::runtime_error("expected the active block to be resident");
    }

    const auto start = std::chrono::steady_clock::now();
    auto restore = cache.prefetch(oldest_id);
    std::size_t decode_ticks = 0;
    do {
        const auto resident = cache.try_get(active_id);
        if (!resident) {
            throw std::runtime_error("an active block was evicted during prefetch");
        }
        ++decode_ticks;
        // The model would dispatch a decode step using resident->data() here.
        std::this_thread::yield();
    } while (restore.wait_for(std::chrono::seconds::zero()) != std::future_status::ready);
    restore.get();
    const auto stop = std::chrono::steady_clock::now();

    const auto restored = cache.try_get(oldest_id);
    if (!restored || cache.checksum(restored).get() != checksum(fixture(oldest_id))) {
        throw std::runtime_error("restored KV bytes differ from the spill");
    }
    const auto stats = cache.stats();
    std::cout << "hot_backend=" << (gpu ? "cuda" : "ram") << " block_bytes=" << page_bytes
              << " direct_io=" << direct << " hot_hits=" << stats.hot_hits
              << " hot_misses=" << stats.hot_misses << " hot_to_host=" << stats.hot_to_host
              << " host_to_disk=" << stats.host_to_disk << " disk_to_host=" << stats.disk_to_host
              << " host_to_hot=" << stats.host_to_hot
              << " overlapping_decode_polls=" << decode_ticks
              << " prefetch_us=" << std::chrono::duration<double, std::micro>(stop - start).count()
              << " checksum_ok=true\n";
} catch (const std::exception& error) {
    std::cerr << "kv_offloader: " << error.what() << '\n';
    return 1;
}
