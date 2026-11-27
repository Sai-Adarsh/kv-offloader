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
#include <system_error>

namespace {

void check_io(int result, const char* message) {
    if (result < 0) {
        throw std::system_error(errno, std::generic_category(), message);
    }
}

off_t file_offset(std::size_t slot, std::size_t bytes) {
    const auto max_offset = static_cast<std::size_t>(std::numeric_limits<off_t>::max());
    if (bytes > max_offset || slot >= max_offset / bytes) {
        throw std::runtime_error("spill file offset exceeded");
    }
    return static_cast<off_t>(slot * bytes);
}

#ifdef OFFLOADER_CUDA
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}
#endif

}  // namespace

HostPage::HostPage(std::size_t bytes) {
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
    : page_bytes_(page_bytes), gpu_(gpu), free_slots_(capacity) {
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
    return slot;
}

void HotStorage::release(std::size_t slot) {
    const std::lock_guard guard(free_mutex_);
    free_slots_.push_back(slot);
}

const std::byte* HotStorage::data(std::size_t slot) const {
    return base_ + slot * page_bytes_;
}

void HotStorage::put(std::size_t slot, const HostPage& host) {
    auto* destination = base_ + slot * page_bytes_;
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
#ifdef OFFLOADER_CUDA
    if (gpu_) {
        check_cuda(cudaMemcpyAsync(cuda_->staging, data(slot), page_bytes_, cudaMemcpyDeviceToHost,
                                   cuda_->transfer));
        check_cuda(cudaStreamSynchronize(cuda_->transfer));
        std::memcpy(host.data(), cuda_->staging, page_bytes_);
        return;
    }
#endif
    std::memcpy(host.data(), data(slot), page_bytes_);
}

DiskStorage::DiskStorage(const std::filesystem::path& directory, std::size_t page_bytes,
                         bool direct)
    : page_bytes_(page_bytes) {
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
    const auto slot = free_slots_.empty() ? next_slot_ : free_slots_.back();
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
    if (free_slots_.empty()) {
        ++next_slot_;
    } else {
        free_slots_.pop_back();
    }
    return slot;
}

void DiskStorage::read(std::size_t slot, HostPage& page) {
    const auto offset = file_offset(slot, page_bytes_);
    ssize_t count = 0;
    do {
        count = pread(fd_, page.data(), page_bytes_, offset);
    } while (count < 0 && errno == EINTR);
    if (count < 0) {
        throw std::system_error(errno, std::generic_category(), "read spill slot");
    }
    if (static_cast<std::size_t>(count) != page_bytes_) {
        throw std::runtime_error("short read from spill slot");
    }
}

void DiskStorage::release(std::size_t slot) { free_slots_.push_back(slot); }
