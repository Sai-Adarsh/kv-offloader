#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

class HostPage {
public:
    explicit HostPage(std::size_t bytes);
    ~HostPage();
    HostPage(const HostPage&) = delete;
    HostPage& operator=(const HostPage&) = delete;
    std::byte* data() const { return data_; }

private:
    std::byte* data_ = nullptr;
};

class HotStorage {
public:
    HotStorage(std::size_t capacity, std::size_t page_bytes, bool gpu);
    ~HotStorage();
    HotStorage(const HotStorage&) = delete;
    HotStorage& operator=(const HotStorage&) = delete;

    std::size_t claim();
    void release(std::size_t slot);
    const std::byte* data(std::size_t slot) const;
    void put(std::size_t slot, const HostPage& host);
    void get(std::size_t slot, HostPage& host);

private:
    std::size_t page_bytes_;
    bool gpu_;
    std::vector<std::byte> ram_;
    std::byte* base_ = nullptr;
#ifdef OFFLOADER_CUDA
    struct Cuda;
    std::unique_ptr<Cuda> cuda_;
#endif
    std::mutex free_mutex_;
    std::vector<std::size_t> free_slots_;
};

class DiskStorage {
public:
    DiskStorage(const std::filesystem::path& directory, std::size_t page_bytes, bool direct);
    ~DiskStorage();
    DiskStorage(const DiskStorage&) = delete;
    DiskStorage& operator=(const DiskStorage&) = delete;

    std::size_t write(const HostPage& page);
    void read(std::size_t slot, HostPage& page);
    void release(std::size_t slot);

private:
    std::size_t page_bytes_;
    std::size_t next_slot_ = 0;
    std::vector<std::size_t> free_slots_;
    int fd_ = -1;
};
