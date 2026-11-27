#pragma once

#include "storage.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <list>
#include <map>
#include <thread>
#include <utility>

class KvCache {
public:
    using BlockId = std::uint64_t;

    class Page {
    public:
        ~Page();
        Page(const Page&) = delete;
        Page& operator=(const Page&) = delete;
        const std::byte* data() const;
        std::size_t slot() const { return slot_; }

    private:
        friend class KvCache;
        explicit Page(std::shared_ptr<HotStorage> storage);

        std::shared_ptr<HotStorage> storage_;
        std::size_t slot_;
    };

    struct Stats {
        std::uint64_t hot_hits;
        std::uint64_t hot_misses;
        std::uint64_t hot_to_host;
        std::uint64_t host_to_disk;
        std::uint64_t disk_to_host;
        std::uint64_t host_to_hot;
        std::uint64_t retired;
        std::size_t disk_pages;
        std::size_t disk_slot_reuses;
        std::size_t disk_high_water_bytes;
        std::uint64_t disk_integrity_failures;
    };

    KvCache(std::size_t hot_capacity, std::size_t host_capacity, std::size_t page_bytes,
            const std::filesystem::path& spill_directory, bool gpu, bool direct,
            std::size_t disk_slots = 64);
    ~KvCache();

    std::future<void> insert(BlockId id, std::vector<std::byte> payload);
    std::future<void> prefetch(BlockId id);
    // Ordered after earlier worker jobs. Existing leases remain valid.
    std::future<bool> retire(BlockId id);
    std::shared_ptr<const Page> try_get(BlockId id);
    std::future<std::uint64_t> checksum(std::shared_ptr<const Page> page);
    Stats stats() const;

private:
    struct HotEntry {
        std::shared_ptr<const Page> page;
        std::uint64_t last_used;
    };

    template<class Result>
    std::future<Result> submit(auto job) {
        std::packaged_task<Result()> task(std::move(job));
        auto result = task.get_future();
        {
            const std::lock_guard guard(jobs_mutex_);
            jobs_.emplace_back(std::move(task));
        }
        jobs_ready_.notify_one();
        return result;
    }

    void run();
    void spill_hot(bool from_disk);
    void spill_host(bool from_disk);
    void install(BlockId id, const HostPage& host, bool from_disk = false);

    std::size_t hot_capacity_;
    std::size_t host_capacity_;
    std::size_t page_bytes_;
    std::size_t disk_slots_;
    std::shared_ptr<HotStorage> hot_storage_;
    DiskStorage disk_storage_;

    std::mutex hot_mutex_;
    std::map<BlockId, HotEntry> hot_;
    std::uint64_t clock_ = 0;

    // Only the I/O worker touches these two tiers.
    std::map<BlockId, std::unique_ptr<HostPage>> host_;
    std::list<BlockId> host_age_;
    std::map<BlockId, std::size_t> disk_;

    std::atomic_uint64_t hot_hits_{0};
    std::atomic_uint64_t hot_misses_{0};
    std::atomic_uint64_t hot_to_host_{0};
    std::atomic_uint64_t host_to_disk_{0};
    std::atomic_uint64_t disk_to_host_{0};
    std::atomic_uint64_t host_to_hot_{0};
    std::atomic_uint64_t retired_{0};
    std::atomic_size_t disk_pages_{0};

    std::mutex jobs_mutex_;
    std::condition_variable jobs_ready_;
    std::deque<std::packaged_task<void()>> jobs_;
    bool stopping_ = false;
    std::thread worker_;
};
