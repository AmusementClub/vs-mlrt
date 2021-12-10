#ifndef VSTRT_CUDA_UTILS_H_
#define VSTRT_CUDA_UTILS_H_

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <cuda_runtime_api.h>

template <typename T, auto deleter>
    requires
        std::default_initializable<T> &&
        std::movable<T> &&
        std::is_trivially_copy_assignable_v<T> &&
        std::convertible_to<T, bool> &&
        std::invocable<decltype(deleter), T>
struct Resource {
    T data;

    [[nodiscard]]
    constexpr Resource() noexcept = default;

    [[nodiscard]]
    constexpr Resource(T && x) noexcept : data(x) {}

    [[nodiscard]]
    constexpr Resource(Resource&& other) noexcept
        : data(std::exchange(other.data, T{}))
    { }

    constexpr Resource& operator=(Resource&& other) noexcept {
        if (this == &other) return *this;
        deleter_(std::move(data));
        data = std::exchange(other.data, T{});
        return *this;
    }

    constexpr Resource& operator=(const Resource & other) = delete;

    Resource(const Resource& other) = delete;

    constexpr operator T() const noexcept {
        return data;
    }

    constexpr auto deleter_(T && x) noexcept {
        if (x) {
            deleter(x);
        }
    }

    constexpr Resource& operator=(T && x) noexcept {
        deleter_(std::move(data));
        data = x;
        return *this;
    }

    constexpr ~Resource() noexcept {
        deleter_(std::move(data));
    }
};

struct MemoryResource {
    Resource<uint8_t *, cudaFreeHost> h_data;
    Resource<uint8_t *, cudaFree> d_data;
    size_t size;
};

using StreamResource = Resource<cudaStream_t, cudaStreamDestroy>;
using GraphExecResource = Resource<cudaGraphExec_t, cudaGraphExecDestroy>;

#endif // VSTRT_CUDA_UTILS_H_
