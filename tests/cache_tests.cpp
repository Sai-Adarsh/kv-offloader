#include "cache.hpp"

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::size_t bytes = 4096;

class TempDir {
public:
    TempDir() {
        auto pattern = (std::filesystem::temp_directory_path() / "kv-cache-test-XXXXXX").string();
        if (!mkdtemp(pattern.data())) throw std::runtime_error("could not create test directory");
        path = std::move(pattern);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
    std::filesystem::path path;
};

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template <class Error, class Action>
void rejects(Action action, std::string_view message) {
    try {
        action();
    } catch (const Error& error) {
        require(std::string_view(error.what()).find(message) != std::string_view::npos, error.what());
        return;
    }
    throw std::runtime_error("operation unexpectedly succeeded");
}

std::vector<std::byte> page(unsigned id) {
    std::vector<std::byte> output(bytes, static_cast<std::byte>(id));
    return output;
}

void contents(const std::shared_ptr<const KvCache::Page>& lease, unsigned id) {
    require(static_cast<bool>(lease), "missing resident lease");
    require(std::all_of(lease->data(), lease->data() + bytes, [=](std::byte value) {
        return value == static_cast<std::byte>(id);
    }), "resident bytes disagree with the inserted page");
}

void slot_budget(const std::filesystem::path& path) {
    HostPage a(bytes), b(bytes);
    std::fill(a.data(), a.data() + bytes, std::byte{42});

    HotStorage hot(1, bytes, false);
    const auto first = hot.claim();
    require(!hot.has_free_slot(), "claimed slot still counted as free");
    rejects<std::logic_error>([&] { hot.claim(); }, "no slot available");
    hot.put(first, a);
    hot.get(first, b);
    require(std::equal(a.data(), a.data() + bytes, b.data()), "hot storage copy");
    hot.release(first);
    require(hot.has_free_slot(), "released slot was not available");

    DiskStorage disk(path, bytes, 2, false);
    rejects<std::runtime_error>([&] { disk.read(0, b); }, "short read");
    const auto slot = disk.write(a);
    const auto second = disk.write(a);
    require(slot != second && disk.high_water_bytes() == 2 * bytes, "disk slot accounting");
    rejects<std::runtime_error>([&] { disk.write(a); }, "budget exhausted");
    disk.read(second, b);
    require(std::equal(a.data(), a.data() + bytes, b.data()), "disk storage copy");
    disk.release(slot);
    require(disk.write(a) == slot && disk.reused_writes() == 1, "disk slot was not reused");

    rejects<std::runtime_error>([&] {
        disk.read(std::numeric_limits<std::size_t>::max(), b);
    }, "offset exceeded");
    rejects<std::invalid_argument>([&] { DiskStorage invalid(path, bytes, 1, false); }, "two nonempty slots");
    rejects<std::invalid_argument>([&] { DiskStorage invalid(path, bytes - 1, 2, true); }, "divisible by 4096");
    rejects<std::system_error>([&] { DiskStorage invalid(path / "missing", bytes, 2, false); }, "create spill file");
    rejects<std::system_error>([] { HostPage impossible(std::numeric_limits<std::size_t>::max()); }, "allocate host page");
}

class FileSizeLimit {
public:
    FileSizeLimit() {
        require(getrlimit(RLIMIT_FSIZE, &saved_) == 0, "read file limit");
        previous_ = std::signal(SIGXFSZ, SIG_IGN);
        require(previous_ != SIG_ERR, "ignore file size signal");
    }
    ~FileSizeLimit() {
        setrlimit(RLIMIT_FSIZE, &saved_);
        std::signal(SIGXFSZ, previous_);
    }
    void use(rlim_t bytes) const {
        auto adjusted = saved_;
        adjusted.rlim_cur = bytes;
        require(setrlimit(RLIMIT_FSIZE, &adjusted) == 0, "set file limit");
    }

private:
    rlimit saved_{};
    void (*previous_)(int) = nullptr;
};

void file_limit(const std::filesystem::path& path) {
    // The process limit forces real pwrite failures on the cache's unlinked file.
    FileSizeLimit limit;
    HostPage page_data(bytes);
    std::fill(page_data.data(), page_data.data() + bytes, std::byte{7});
    DiskStorage disk(path, bytes, 2, false);
    limit.use(0);
    rejects<std::system_error>([&] { disk.write(page_data); }, "write spill slot");
    require(disk.high_water_bytes() == 0, "failed write consumed a slot");
    limit.use(bytes / 2);
    rejects<std::runtime_error>([&] { disk.write(page_data); }, "short write");
    require(disk.high_water_bytes() == 0, "short write consumed a slot");
    limit.use(bytes);
    require(disk.write(page_data) == 0, "corrected retry did not reclaim the first slot");
}

class ScopedFd {
public:
    explicit ScopedFd(int fd) : fd_(fd) { require(fd != -1, "open test descriptor"); }
    ~ScopedFd() { close(fd_); }
    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    int get() const { return fd_; }

private:
    int fd_;
};

void unreadable_spill(const std::filesystem::path& path) {
    // There are no cache worker threads yet. Reserve and release the next free
    // descriptor so we can operate on the anonymous file without a production hook.
    const ScopedFd write_only(open("/dev/null", O_WRONLY));
    const auto fd = dup(write_only.get());
    require(fd >= 0 && close(fd) == 0, "identify next temporary file descriptor");
    DiskStorage disk(path, bytes, 2, false);
    HostPage source(bytes), target(bytes);
    std::fill(source.data(), source.data() + bytes, std::byte{11});
    const auto slot = disk.write(source);
    struct stat backing{};
    require(fstat(fd, &backing) == 0 && S_ISREG(backing.st_mode) && backing.st_nlink == 0
            && backing.st_size == bytes, "descriptor is not the anonymous spill file");

    require(ftruncate(fd, bytes / 2) == 0, "truncate a written page");
    rejects<std::runtime_error>([&] { disk.read(slot, target); }, "short read");

    // Substitute a write-only descriptor briefly to make pread report EBADF.
    const ScopedFd backup(dup(fd));
    require(dup2(write_only.get(), fd) == fd, "install unreadable descriptor");
    rejects<std::system_error>([&] { disk.read(slot, target); }, "read spill slot");
    require(dup2(backup.get(), fd) == fd, "restore original backing descriptor");

    disk.release(slot);
    require(disk.write(source) == slot, "read failure prevented reclaiming the slot");
    disk.read(slot, target);
    require(std::equal(source.data(), source.data() + bytes, target.data()), "replacement page was corrupted");
}

void failed_spill_promotion(const std::filesystem::path& path) {
    KvCache cache(1, 1, bytes, path, false, false, 2);
    cache.insert(1, page(1)).get();
    cache.insert(2, page(2)).get();
    cache.insert(3, page(3)).get();
    require(cache.stats().disk_pages == 1, "expected a full logical disk tier");
    {
        FileSizeLimit limit;
        limit.use(bytes + bytes / 2);
        rejects<std::runtime_error>([&] { cache.prefetch(1).get(); }, "short write");
    }
    require(cache.stats().disk_pages == 1 && cache.stats().disk_high_water_bytes == bytes,
            "failed disk promotion consumed the scratch slot");
    contents(cache.try_get(3), 3);
    require(!cache.try_get(1), "failed promotion published data");
    cache.prefetch(1).get();
    contents(cache.try_get(1), 1);
    require(cache.stats().disk_pages == 1 && cache.stats().disk_high_water_bytes == 2 * bytes,
            "retry did not use the scratch slot once and reclaim the source");
    cache.prefetch(2).get();
    contents(cache.try_get(2), 2);
    cache.prefetch(3).get();
    contents(cache.try_get(3), 3);
}

void lifecycle(const std::filesystem::path& path) {
    rejects<std::invalid_argument>([&] { KvCache bad(0, 1, bytes, path, false, false); }, "positive");
    rejects<std::invalid_argument>([&] {
        KvCache bad(std::numeric_limits<std::size_t>::max(), 1, bytes, path, false, false);
    }, "staging slot");
    rejects<std::invalid_argument>([&] {
        KvCache bad(1, 1, std::numeric_limits<std::size_t>::max(), path, false, false);
    }, "exceed address space");

    KvCache cache(1, 1, bytes, path, false, false, 2);
    rejects<std::invalid_argument>([&] { cache.insert(9, {}); }, "exactly one page");
    rejects<std::invalid_argument>([&] { cache.checksum(nullptr); }, "resident block lease");
    rejects<std::invalid_argument>([&] { cache.prefetch(9).get(); }, "unknown KV block");
    require(!cache.retire(9).get(), "retired an absent page");

    cache.insert(1, page(1)).get();
    cache.prefetch(1).get();
    cache.insert(2, page(2)).get();
    auto active = cache.try_get(2);
    contents(active, 2);
    rejects<std::invalid_argument>([&] { cache.insert(1, page(1)).get(); }, "already exists");
    rejects<std::runtime_error>([&] { cache.prefetch(1).get(); }, "active leases");
    require(cache.retire(2).get(), "retire failed");
    require(!cache.try_get(2), "retired index unexpectedly allowed a new lease");
    contents(active, 2);
    rejects<std::runtime_error>([&] { cache.prefetch(1).get(); }, "active leases");
    active.reset();
    cache.prefetch(1).get();
    contents(cache.try_get(1), 1);

    {
        KvCache foreign(1, 1, bytes, path, false, false, 2);
        foreign.insert(9, page(9)).get();
        rejects<std::invalid_argument>([&] { cache.checksum(foreign.try_get(9)); }, "lease from this cache");
    }

    cache.insert(3, page(3)).get();
    cache.insert(4, page(4)).get();
    require(cache.stats().disk_pages == 1, "expected one normal spill");
    rejects<std::runtime_error>([&] { cache.insert(5, page(5)).get(); }, "slot budget exhausted");
    auto pinned = cache.try_get(4);
    rejects<std::runtime_error>([&] { cache.prefetch(1).get(); }, "active leases");
    require(cache.stats().disk_high_water_bytes == bytes, "failed promotion consumed scratch");
    pinned.reset();

    auto restore = cache.prefetch(1);
    auto retire = cache.retire(1);
    restore.get();
    require(retire.get() && !cache.try_get(1), "retire submitted behind in-flight restore");
    require(cache.stats().disk_high_water_bytes == 2 * bytes, "full-tier swap ignored scratch budget");
    require(cache.retire(3).get(), "retire from disk failed");
    cache.insert(5, page(5)).get();
    require(cache.retire(4).get(), "retire from warm failed after owner replacement");
    contents(cache.try_get(5), 5);
}

}  // namespace

int main() try {
    TempDir temp;
    slot_budget(temp.path);
    file_limit(temp.path);
    unreadable_spill(temp.path);
    failed_spill_promotion(temp.path);
    lifecycle(temp.path);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
