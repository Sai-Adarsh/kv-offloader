#include "cache.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

std::size_t nonzero(std::size_t value) {
    if (value == 0) {
        throw std::invalid_argument("cache capacities and page size must be positive");
    }
    return value;
}

std::size_t checked_page_bytes(std::size_t bytes, std::size_t capacity) {
    nonzero(bytes);
    if (bytes > std::numeric_limits<std::size_t>::max() / capacity) {
        throw std::invalid_argument("hot tier allocation would exceed address space");
    }
    return bytes;
}

constexpr std::uint64_t checksum_start = 14695981039346656037ull;
constexpr std::uint64_t checksum_prime = 1099511628211ull;

std::uint64_t update_checksum(std::uint64_t sum, std::byte value) {
    return (sum ^ std::to_integer<unsigned char>(value)) * checksum_prime;
}

}  // namespace

KvCache::Page::Page(std::shared_ptr<HotStorage> storage)
    : storage_(std::move(storage)), slot_(storage_->claim()) {}

KvCache::Page::~Page() { storage_->release(slot_); }

KvCache::KvCache(std::size_t hot_capacity, std::size_t host_capacity, std::size_t page_bytes,
                 const std::filesystem::path& spill_directory, bool gpu, bool direct, std::size_t disk_slots)
    : hot_capacity_(nonzero(hot_capacity)), host_capacity_(nonzero(host_capacity)),
      page_bytes_(checked_page_bytes(page_bytes, hot_capacity_)), disk_slots_(disk_slots),
      hot_storage_(std::make_shared<HotStorage>(hot_capacity_, page_bytes_, gpu)),
      disk_storage_(spill_directory, page_bytes_, disk_slots_, direct), worker_([this] { run(); }) {}

KvCache::~KvCache() {
    {
        const std::lock_guard guard(jobs_mutex_);
        stopping_ = true;
    }
    jobs_ready_.notify_one();
    worker_.join();
}

void KvCache::run() {
    for (;;) {
        std::packaged_task<void()> job;
        {
            std::unique_lock lock(jobs_mutex_);
            jobs_ready_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
            if (jobs_.empty()) {
                return;
            }
            job = std::move(jobs_.front());
            jobs_.pop_front();
        }
        job();
    }
}

std::future<void> KvCache::insert(BlockId id, std::vector<std::byte> payload) {
    if (payload.size() != page_bytes_) {
        throw std::invalid_argument("KV payload must contain exactly one page");
    }
    return submit<void>([this, id, payload = std::move(payload)] {
        {
            const std::lock_guard guard(hot_mutex_);
            if (hot_.contains(id) || host_.contains(id) || disk_.contains(id)) {
                throw std::invalid_argument("block already exists");
            }
        }
        HostPage host(page_bytes_);
        std::memcpy(host.data(), payload.data(), page_bytes_);
        install(id, host);
    });
}

std::future<void> KvCache::prefetch(BlockId id) {
    return submit<void>([this, id] {
        {
            const std::lock_guard guard(hot_mutex_);
            const auto already_hot = hot_.find(id);
            if (already_hot != hot_.end()) {
                already_hot->second.last_used = ++clock_;
                return;
            }
        }
        if (const auto resident = host_.find(id); resident != host_.end()) {
            // The incoming block is already holding one host page. Remove it from
            // the eviction queue while making a slot for the incoming block.
            auto page = host_.extract(resident);
            host_age_.erase(std::find(host_age_.begin(), host_age_.end(), id));
            try {
                install(id, *page.mapped());
            } catch (...) {
                host_age_.push_front(id);
                host_.insert(std::move(page));
                throw;
            }
            return;
        }
        if (const auto spilled = disk_.find(id); spilled != disk_.end()) {
            HostPage staging(page_bytes_);
            disk_storage_.read(spilled->second, staging);
            ++disk_to_host_;
            // Retain the old slot until installation succeeds. An extra spill
            // can use the reserved scratch slot without clobbering this page.
            install(id, staging, true);
            disk_storage_.release(spilled->second);
            disk_.erase(spilled);
            --disk_pages_;
            return;
        }
        throw std::invalid_argument("unknown KV block");
    });
}

std::future<bool> KvCache::retire(BlockId id) {
    return submit<bool>([this, id] {
        std::map<BlockId, HotEntry>::node_type resident;
        {
            const std::lock_guard guard(hot_mutex_);
            resident = hot_.extract(id);
        }
        if (!resident.empty()) {
            // Removing the index stops new leases. The slot returns to the
            // allocator only when the worker and all existing leases let go.
            ++retired_;
            return true;
        }
        if (const auto warm = host_.find(id); warm != host_.end()) {
            host_age_.erase(std::find(host_age_.begin(), host_age_.end(), id));
            host_.erase(warm);
        } else if (const auto spilled = disk_.find(id); spilled != disk_.end()) {
            disk_storage_.release(spilled->second);
            disk_.erase(spilled);
            --disk_pages_;
        } else {
            return false;
        }
        ++retired_;
        return true;
    });
}

std::shared_ptr<const KvCache::Page> KvCache::try_get(BlockId id) {
    const std::lock_guard guard(hot_mutex_);
    const auto hit = hot_.find(id);
    if (hit == hot_.end()) {
        ++hot_misses_;
        return {};
    }
    ++hot_hits_;
    hit->second.last_used = ++clock_;
    return hit->second.page;
}

std::future<std::uint64_t> KvCache::checksum(std::shared_ptr<const Page> page) {
    if (!page) {
        throw std::invalid_argument("checksum requires a resident block lease");
    }
    return submit<std::uint64_t>([this, page = std::move(page)] {
        HostPage host(page_bytes_);
        hot_storage_->get(page->slot(), host);
        std::uint64_t sum = checksum_start;
        for (std::size_t i = 0; i < page_bytes_; ++i) {
            sum = update_checksum(sum, host.data()[i]);
        }
        return sum;
    });
}

KvCache::Stats KvCache::stats() const {
    return {hot_hits_, hot_misses_, hot_to_host_, host_to_disk_, disk_to_host_, host_to_hot_,
            retired_, disk_pages_, disk_storage_.reused_writes(), disk_storage_.high_water_bytes()};
}

void KvCache::spill_hot(bool from_disk) {
    // A retired resident may still occupy its physical slot through a lease.
    if (hot_storage_->has_free_slot()) return;
    std::map<BlockId, HotEntry>::node_type victim;
    {
        const std::lock_guard guard(hot_mutex_);
        auto oldest = hot_.end();
        for (auto current = hot_.begin(); current != hot_.end(); ++current) {
            if (current->second.page.use_count() == 1 &&
                (oldest == hot_.end() || current->second.last_used < oldest->second.last_used)) {
                oldest = current;
            }
        }
        if (oldest == hot_.end()) {
            // A lease may have ended after the initial capacity check.
            if (hot_storage_->has_free_slot()) return;
            throw std::runtime_error("all hot slots have active leases");
        }
        victim = hot_.extract(oldest);
    }

    try {
        spill_host(from_disk);
        auto page = std::make_unique<HostPage>(page_bytes_);
        hot_storage_->get(victim.mapped().page->slot(), *page);
        host_age_.push_back(victim.key());
        try {
            host_.try_emplace(victim.key(), std::move(page));
        } catch (...) {
            host_age_.pop_back();
            throw;
        }
        ++hot_to_host_;
    } catch (...) {
        const std::lock_guard guard(hot_mutex_);
        hot_.insert(std::move(victim));
        throw;
    }
}

void KvCache::spill_host(bool from_disk) {
    if (host_.size() < host_capacity_) {
        return;
    }
    if (!from_disk && disk_.size() >= disk_slots_ - 1) {
        throw std::runtime_error("spill file slot budget exhausted; retire an inactive block");
    }
    const auto victim = host_age_.front();
    const auto slot = disk_storage_.write(*host_.at(victim));
    try {
        disk_.try_emplace(victim, slot);
    } catch (...) {
        disk_storage_.release(slot);
        throw;
    }
    host_age_.pop_front();
    host_.erase(victim);
    ++disk_pages_;
    ++host_to_disk_;
}

void KvCache::install(BlockId id, const HostPage& host, bool from_disk) {
    spill_hot(from_disk);
    std::shared_ptr<const Page> page(new Page(hot_storage_));
    hot_storage_->put(page->slot(), host);
    const std::lock_guard guard(hot_mutex_);
    hot_.try_emplace(id, HotEntry{std::move(page), ++clock_});
    ++host_to_hot_;
}
