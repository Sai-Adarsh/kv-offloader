#include "cache.hpp"

#include <chrono>
#include <charconv>
#include <deque>
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
    auto state = id;
    for (std::size_t i = 0; i < page_bytes; i += 8) {
        state += 0x9e3779b97f4a7c15ULL;
        auto value = state;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        value ^= value >> 31;
        for (int byte = 0; byte < 8; ++byte) {
            result[i + byte] = static_cast<std::byte>((value >> (byte * 8)) & 0xff);
        }
    }
    return result;
}

std::size_t count(std::string_view text, std::size_t minimum, std::size_t maximum) {
    std::size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()
        || value < minimum || value > maximum) {
        throw std::invalid_argument("numeric option outside its supported range");
    }
    return value;
}

void churn(KvCache& cache, std::size_t disk_slots, std::size_t cycles) {
    const auto live_limit = 2 + 2 + disk_slots - 1;
    std::deque<KvCache::BlockId> active;
    KvCache::BlockId next = 0;
    const auto insert = [&] {
        cache.insert(next, fixture(next)).get();
        active.push_back(next++);
    };
    while (active.size() < live_limit) insert();

    try {
        cache.insert(next, fixture(next)).get();
        throw std::logic_error("cache exceeded its normal slot budget");
    } catch (const std::runtime_error& error) {
        if (!std::string_view(error.what()).starts_with("spill file slot budget exhausted")) throw;
    }

    // Retire a resident while an existing reader still holds it.
    const auto newest = active.back();
    auto pinned = cache.try_get(newest);
    if (!pinned || !cache.retire(newest).get() || cache.try_get(newest)
        || cache.checksum(pinned).get() != page_checksum(fixture(newest))) {
        throw std::runtime_error("retirement did not preserve an existing lease");
    }
    pinned.reset();
    active.pop_back();

    // Queue retirement before the older, in-flight restore has completed.
    auto restore = cache.prefetch(active.front());
    auto retire = cache.retire(active.front());
    restore.get();
    if (!retire.get() || cache.try_get(active.front())) {
        throw std::runtime_error("retirement did not follow an earlier prefetch");
    }
    active.pop_front();
    while (active.size() < live_limit) insert();

    const auto ready = std::chrono::steady_clock::now();
    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        const auto id = active.front();
        cache.prefetch(id).get();
        auto lease = cache.try_get(id);
        if (!lease || cache.checksum(lease).get() != page_checksum(fixture(id)) || !cache.retire(id).get()) {
            throw std::runtime_error("incorrect retired page during churn");
        }
        lease.reset();
        active.pop_front();
        insert();
    }
    const auto elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - ready).count();
    const auto stats = cache.stats();
    if (stats.disk_high_water_bytes > disk_slots * page_bytes || stats.disk_slot_reuses == 0) {
        throw std::runtime_error("spill slots did not remain bounded and reusable");
    }
    std::cout << "cycles=" << cycles << " steady_live_blocks=" << active.size()
              << " retired=" << stats.retired << " disk_pages=" << stats.disk_pages
              << " disk_slot_reuses=" << stats.disk_slot_reuses
              << " disk_high_water_bytes=" << stats.disk_high_water_bytes
              << " disk_budget_bytes=" << disk_slots * page_bytes
              << " disk_integrity_failures=" << stats.disk_integrity_failures
              << " elapsed_ms=" << elapsed_ms
              << " checksums_ok=true\n";
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: kv_offloader SPILL_DIRECTORY [--gpu] [--direct] [--disk-slots N] [--churn N]");
    }
    bool gpu = false;
    bool direct = false;
    std::size_t disk_slots = 4;
    std::size_t cycles = 0;
    for (int i = 2; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--gpu") {
            gpu = true;
        } else if (flag == "--direct") {
            direct = true;
        } else if (flag == "--disk-slots" && i + 1 < argc) {
            disk_slots = count(argv[++i], 2, 4096);
        } else if (flag == "--churn" && i + 1 < argc) {
            cycles = count(argv[++i], 1, 100000);
        } else {
            throw std::invalid_argument("unknown kv_offloader flag");
        }
    }

    KvCache cache(2, 2, page_bytes, std::filesystem::path(argv[1]), gpu, direct, disk_slots);
    if (cycles > 0) {
        churn(cache, disk_slots, cycles);
        return 0;
    }
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
    std::size_t resident_polls = 0;
    do {
        const auto resident = cache.try_get(active_id);
        if (!resident) {
            throw std::runtime_error("an active block was evicted during prefetch");
        }
        ++resident_polls;
        std::this_thread::yield();
    } while (restore.wait_for(std::chrono::seconds::zero()) != std::future_status::ready);
    restore.get();
    const auto stop = std::chrono::steady_clock::now();

    const auto restored = cache.try_get(oldest_id);
    if (!restored || cache.checksum(restored).get() != page_checksum(fixture(oldest_id))) {
        throw std::runtime_error("restored KV bytes differ from the spill");
    }
    const auto stats = cache.stats();
    std::cout << "hot_backend=" << (gpu ? "cuda" : "ram") << " block_bytes=" << page_bytes
              << " direct_io=" << direct << " hot_hits=" << stats.hot_hits
              << " hot_misses=" << stats.hot_misses << " hot_to_host=" << stats.hot_to_host
              << " host_to_disk=" << stats.host_to_disk << " disk_to_host=" << stats.disk_to_host
              << " host_to_hot=" << stats.host_to_hot
              << " disk_integrity_failures=" << stats.disk_integrity_failures
              << " resident_polls_during_restore=" << resident_polls
              << " prefetch_us=" << std::chrono::duration<double, std::micro>(stop - start).count()
              << " checksum_ok=true\n";
} catch (const std::exception& error) {
    std::cerr << "kv_offloader: " << error.what() << '\n';
    return 1;
}
