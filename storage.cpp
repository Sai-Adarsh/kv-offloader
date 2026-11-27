#include "storage.hpp"

#ifdef OFFLOADER_CUDA
#include <cuda_runtime_api.h>
#endif

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

void check_io(int result, const char* message) {
    if (result < 0) {
        throw std::system_error(errno, std::generic_category(), message);
    }
}

void check_page_size(const HostPage& page, std::size_t page_bytes) {
    if (page.size() != page_bytes) {
        throw std::invalid_argument("host page size does not match storage page size");
    }
}

off_t file_offset(std::size_t slot, std::size_t bytes) {
    const auto max_offset = static_cast<std::size_t>(std::numeric_limits<off_t>::max());
    if (bytes > max_offset || slot >= max_offset / bytes) {
        throw std::runtime_error("spill file offset exceeded");
    }
    return static_cast<off_t>(slot * bytes);
}

std::size_t checked_hot_page_bytes(std::size_t capacity, std::size_t bytes) {
    if (capacity == 0 || bytes == 0) {
        throw std::invalid_argument("hot capacity and page size must be positive");
    }
    if (bytes > std::numeric_limits<std::size_t>::max() / capacity) {
        throw std::invalid_argument("hot tier allocation would exceed address space");
    }
    return bytes;
}

#ifdef OFFLOADER_CUDA
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}
#endif

}  // namespace

std::uint64_t page_checksum(std::span<const std::byte> bytes) {
    std::uint64_t sum = 14695981039346656037ull;
    for (const std::byte byte : bytes) {
        sum = (sum ^ std::to_integer<unsigned char>(byte)) * 1099511628211ull;
    }
    return sum;
}

HostPage::HostPage(std::size_t bytes) : bytes_(bytes) {
    void* allocation = nullptr;
    const int status = posix_memalign(&allocation, 4096, bytes);
    if (status != 0) {
        throw std::system_error(status, std::generic_category(), "allocate host page");
    }
    data_ = static_cast<std::byte*>(allocation);
}

HostPage::~HostPage() { std::free(data_); }

#ifdef OFFLOADER_CUDA
struct HotStorage::Cuda {
    Cuda(std::size_t bytes, std::size_t page_bytes) {
        void* allocation = nullptr;
        check_cuda(cudaMalloc(&allocation, bytes));
        data = static_cast<std::byte*>(allocation);
        try {
            check_cuda(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
            check_cuda(cudaMallocHost(&staging, page_bytes));
        } catch (...) {
            if (transfer) {
                cudaStreamDestroy(transfer);
            }
            cudaFree(data);
            throw;
        }
    }

    ~Cuda() {
        cudaStreamDestroy(transfer);
        cudaFreeHost(staging);
        cudaFree(data);
    }

    std::byte* data = nullptr;
    void* staging = nullptr;
    cudaStream_t transfer = nullptr;
};
#endif

HotStorage::HotStorage(std::size_t capacity, std::size_t page_bytes, bool gpu)
    : page_bytes_(checked_hot_page_bytes(capacity, page_bytes)), gpu_(gpu),
      free_slots_(capacity), claimed_(capacity) {
    std::iota(free_slots_.begin(), free_slots_.end(), 0);
    if (gpu_) {
#ifdef OFFLOADER_CUDA
        cuda_ = std::make_unique<Cuda>(capacity * page_bytes_, page_bytes_);
        base_ = cuda_->data;
#else
        throw std::invalid_argument("rebuild with ENABLE_CUDA=ON to use the GPU hot tier");
#endif
    } else {
        ram_.resize(capacity * page_bytes_);
        base_ = ram_.data();
    }
}

HotStorage::~HotStorage() = default;

std::size_t HotStorage::claim() {
    const std::lock_guard guard(free_mutex_);
    if (free_slots_.empty()) {
        throw std::logic_error("no slot available in the hot tier");
    }
    const auto slot = free_slots_.back();
    free_slots_.pop_back();
    claimed_[slot].store(true, std::memory_order_relaxed);
    return slot;
}

void HotStorage::release(std::size_t slot) {
    const std::lock_guard guard(free_mutex_);
    if (slot >= claimed_.size() || !claimed_[slot].load(std::memory_order_relaxed)) {
        throw std::logic_error("cannot release an inactive hot slot");
    }
    claimed_[slot].store(false, std::memory_order_relaxed);
    free_slots_.push_back(slot);
}

bool HotStorage::has_free_slot() {
    const std::lock_guard guard(free_mutex_);
    return !free_slots_.empty();
}

std::byte* HotStorage::address(std::size_t slot) const {
    if (slot >= claimed_.size()) {
        throw std::out_of_range("hot slot is out of range");
    }
    // A lease keeps the slot live. Checking it here does not take the allocator mutex.
    if (!claimed_[slot].load(std::memory_order_relaxed)) {
        throw std::logic_error("cannot access an inactive hot slot");
    }
    return base_ + slot * page_bytes_;
}

const std::byte* HotStorage::data(std::size_t slot) const { return address(slot); }

void HotStorage::put(std::size_t slot, const HostPage& host) {
    check_page_size(host, page_bytes_);
    auto* destination = address(slot);
#ifdef OFFLOADER_CUDA
    if (gpu_) {
        std::memcpy(cuda_->staging, host.data(), page_bytes_);
        check_cuda(cudaMemcpyAsync(destination, cuda_->staging, page_bytes_, cudaMemcpyHostToDevice,
                                   cuda_->transfer));
        check_cuda(cudaStreamSynchronize(cuda_->transfer));
        return;
    }
#endif
    std::memcpy(destination, host.data(), page_bytes_);
}

void HotStorage::get(std::size_t slot, HostPage& host) {
    check_page_size(host, page_bytes_);
    const auto* source = address(slot);
#ifdef OFFLOADER_CUDA
    if (gpu_) {
        check_cuda(cudaMemcpyAsync(cuda_->staging, source, page_bytes_, cudaMemcpyDeviceToHost,
                                   cuda_->transfer));
        check_cuda(cudaStreamSynchronize(cuda_->transfer));
        std::memcpy(host.data(), cuda_->staging, page_bytes_);
        return;
    }
#endif
    std::memcpy(host.data(), source, page_bytes_);
}

DiskStorage::DiskStorage(const std::filesystem::path& directory, std::size_t page_bytes,
                         std::size_t slots, bool direct)
    : page_bytes_(page_bytes) {
    if (slots < 2 || page_bytes == 0) {
        throw std::invalid_argument("the spill file needs at least two nonempty slots");
    }
    (void)file_offset(slots - 1, page_bytes);
    checksums_.resize(slots);
    free_slots_.reserve(slots);
    auto pattern = (directory / "kv-offloader-XXXXXX").string();
    fd_ = mkstemp(pattern.data());
    check_io(fd_, "create spill file");
    try {
        check_io(unlink(pattern.c_str()), "unlink spill file");
        if (direct) {
            if (page_bytes_ % 4096 != 0) {
                throw std::invalid_argument("direct spill I/O requires page sizes divisible by 4096");
            }
#ifdef __linux__
            check_io(fcntl(fd_, F_SETFL, O_DIRECT), "enable O_DIRECT");
#elif defined(__APPLE__)
            check_io(fcntl(fd_, F_NOCACHE, 1), "enable F_NOCACHE");
#else
            throw std::invalid_argument("direct spill I/O requires Linux or macOS");
#endif
        }
    } catch (...) {
        close(fd_);
        throw;
    }
}

DiskStorage::~DiskStorage() { close(fd_); }

std::size_t DiskStorage::write(const HostPage& page) {
    check_page_size(page, page_bytes_);
    const bool reuse = !free_slots_.empty();
    if (!reuse && next_slot_.load() == checksums_.size()) {
        throw std::runtime_error("spill file slot budget exhausted");
    }
    const auto slot = reuse ? free_slots_.back() : next_slot_.load();
    const auto offset = file_offset(slot, page_bytes_);
    ssize_t count = 0;
    do {
        count = pwrite(fd_, page.data(), page_bytes_, offset);
    } while (count < 0 && errno == EINTR);
    if (count < 0) {
        throw std::system_error(errno, std::generic_category(), "write spill slot");
    }
    if (static_cast<std::size_t>(count) != page_bytes_) {
        throw std::runtime_error("short write to spill slot");
    }
    checksums_[slot] = page_checksum({page.data(), page_bytes_});
    if (reuse) {
        free_slots_.pop_back();
        ++reused_writes_;
    } else {
        ++next_slot_;
    }
    return slot;
}

void DiskStorage::read(std::size_t slot, HostPage& page) {
    check_page_size(page, page_bytes_);
    const auto offset = file_offset(slot, page_bytes_);
    if (slot >= checksums_.size() || !checksums_[slot]) {
        throw std::runtime_error("spill slot has no live page");
    }
    ssize_t count = 0;
    do {
        count = pread(fd_, page.data(), page_bytes_, offset);
    } while (count < 0 && errno == EINTR);
    if (count < 0) {
        throw std::system_error(errno, std::generic_category(), "read spill slot");
    }
    if (static_cast<std::size_t>(count) != page_bytes_) {
        ++integrity_failures_;
        throw std::runtime_error("short read from spill slot");
    }
    if (page_checksum({page.data(), page_bytes_}) != *checksums_[slot]) {
        ++integrity_failures_;
        throw std::runtime_error("spill page checksum mismatch at slot " + std::to_string(slot));
    }
}

void DiskStorage::release(std::size_t slot) {
    if (slot >= checksums_.size() || !checksums_[slot]) {
        throw std::logic_error("cannot release an inactive spill slot");
    }
    checksums_[slot].reset();
    free_slots_.push_back(slot);
}
