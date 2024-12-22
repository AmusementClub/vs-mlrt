#include <array>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <hip/hip_runtime.h>

#include <migraphx/migraphx.h>

#if __has_include(<migraphx/version.h>)
#include <migraphx/version.h>
#else
#define NO_MIGX_VERSION
#endif

#include "config.h"

using namespace std::string_literals;

#define checkError(expr) do {                                                  \
    using namespace std::string_literals;                                      \
    migraphx_status __err = expr;                                              \
    if (__err != migraphx_status_success) {                                    \
        const char * message = getErrorString(__err);                          \
        return set_error("'"s + # expr + "' failed: " + message);              \
    }                                                                          \
} while(0)

#define checkHIPError(expr) do {                                               \
    using namespace std::string_literals;                                      \
    hipError_t __err = expr;                                                   \
    if (__err != hipSuccess) {                                                 \
        const char * message = hipGetErrorString(__err);                       \
        return set_error("'"s + # expr + "' failed: " + message);              \
    }                                                                          \
} while(0)

static const VSPlugin * myself = nullptr;

static inline const char * getErrorString(migraphx_status status) {
    switch (status) {
        case migraphx_status_success:
            return "success";
        case migraphx_status_bad_param:
            return "bad param";
        case migraphx_status_unknown_target:
            return "unknown target";
        case migraphx_status_unknown_error:
            return "unknown error";
        default:
            return "undefined error";
    }
}

static void setDimensions(
    std::unique_ptr<VSVideoInfo> & vi,
    const std::array<int, 4> & input_shape,
    const std::array<int, 4> & output_shape,
    int bitsPerSample,
    VSCore * core,
    const VSAPI * vsapi,
    bool flexible_output
) noexcept {

    vi->height *= output_shape[2] / input_shape[2];
    vi->width *= output_shape[3] / input_shape[3];

    if (output_shape[1] == 1 || flexible_output) {
        vi->format = vsapi->registerFormat(cmGray, stFloat, bitsPerSample, 0, 0, core);
    } else if (output_shape[1] == 3) {
        vi->format = vsapi->registerFormat(cmRGB, stFloat, bitsPerSample, 0, 0, core);
    }
}

static inline
std::optional<std::string> checkNodes(
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    for (const auto & vi : vis) {
        if (!isConstantFormat(vi)) {
            return "video format must be constant";
        }

        if (vi->width != vis[0]->width || vi->height != vis[0]->height) {
            return "dimensions of clips mismatch";
        }

        if (vi->numFrames != vis[0]->numFrames) {
            return "number of frames mismatch";
        }

        if (vi->format->subSamplingH != 0 || vi->format->subSamplingW != 0) {
            return "clip must not be sub-sampled";
        }

        if (vi->format->sampleType != vis[0]->format->sampleType) {
            return "sample type mismatch";
        }

        if (vi->format->bitsPerSample != vis[0]->format->bitsPerSample) {
            return "bits per sample mismatch";
        }
    }

    return {};
}

static inline
int numPlanes(
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    int num_planes = 0;

    for (const auto & vi : vis) {
        num_planes += vi->format->numPlanes;
    }

    return num_planes;
}

// 0: integer, 1: float, -1: unknown
static inline
int getSampleType(migraphx_shape_datatype_t type) noexcept {
    switch (type) {
        case migraphx_shape_uint8_type:
        case migraphx_shape_uint16_type:
        case migraphx_shape_uint32_type:
        case migraphx_shape_uint64_type:
            return 0;
        case migraphx_shape_half_type:
        case migraphx_shape_float_type:
        case migraphx_shape_double_type:
            return 1;
        default:
            return -1;
    }
}

static inline
int getBytesPerSample(migraphx_shape_datatype_t type) noexcept {
    switch (type) {
        case migraphx_shape_uint8_type:
            return 1;
        case migraphx_shape_half_type:
        case migraphx_shape_uint16_type:
            return 2;
        case migraphx_shape_float_type:
        case migraphx_shape_uint32_type:
            return 4;
        case migraphx_shape_double_type:
        case migraphx_shape_uint64_type:
            return 8;
        default:
            return 0;
    }
}

static inline void VS_CC getDeviceProp(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) {

    int err;
    int device_id = static_cast<int>(vsapi->propGetInt(in, "device_id", 0, &err));
    if (err) {
        device_id = 0;
    }

    hipDeviceProp_t prop;
    if (auto err = hipGetDeviceProperties(&prop, device_id); err != hipSuccess) {
        vsapi->setError(out, hipGetErrorString(err));
        return ;
    }

    auto setProp = [&](const char * name, auto value, int data_length = -1) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, int>) {
            vsapi->propSetInt(out, name, value, paReplace);
        } else if constexpr (std::is_same_v<T, size_t>) {
            vsapi->propSetInt(out, name, static_cast<int64_t>(value), paReplace);
        } else if constexpr (std::is_same_v<T, char *>) {
            vsapi->propSetData(out, name, value, data_length, paReplace);
        }
    };

    int driver_version;
    if (auto err = hipDriverGetVersion(&driver_version); err != hipSuccess) {
        vsapi->setError(out, hipGetErrorString(err));
        return ;
    }
    setProp("driver_version", driver_version);

    setProp("name", prop.name);
    {
        std::array<int64_t, 16> uuid;
        for (int i = 0; i < 16; ++i) {
            uuid[i] = prop.uuid.bytes[i];
        }
        vsapi->propSetIntArray(out, "uuid", std::data(uuid), std::size(uuid));
    }
    setProp("total_global_memory", prop.totalGlobalMem);
    setProp("shared_memory_per_block", prop.sharedMemPerBlock);
    setProp("regs_per_block", prop.regsPerBlock);
    setProp("warp_size", prop.warpSize);
    setProp("mem_pitch", prop.memPitch);
    setProp("max_threads_per_block", prop.maxThreadsPerBlock);
    setProp("clock_rate", prop.clockRate);
    setProp("total_const_mem", prop.totalConstMem);
    setProp("major", prop.major);
    setProp("minor", prop.minor);
    setProp("texture_alignment", prop.textureAlignment);
    setProp("texture_pitch_alignment", prop.texturePitchAlignment);
    setProp("device_overlap", prop.deviceOverlap);
    setProp("multi_processor_count", prop.multiProcessorCount);
    setProp("kernel_exec_timeout_enabled", prop.kernelExecTimeoutEnabled);
    setProp("integrated", prop.integrated);
    setProp("can_map_host_memory", prop.canMapHostMemory);
    setProp("compute_mode", prop.computeMode);
    setProp("concurrent_kernels", prop.concurrentKernels);
    setProp("ecc_enabled", prop.ECCEnabled);
    setProp("pci_bus_id", prop.pciBusID);
    setProp("pci_device_id", prop.pciDeviceID);
    setProp("pci_domain_id", prop.pciDomainID);
    setProp("tcc_driver", prop.tccDriver);
    setProp("async_engine_count", prop.asyncEngineCount);
    setProp("unified_addressing", prop.unifiedAddressing);
    setProp("memory_clock_rate", prop.memoryClockRate);
    setProp("memory_bus_width", prop.memoryBusWidth);
    setProp("l2_cache_size", prop.l2CacheSize);
    setProp("persisting_l2_cache_max_size", prop.persistingL2CacheMaxSize);
    setProp("max_threads_per_multiprocessor", prop.maxThreadsPerMultiProcessor);
    setProp("stream_priorities_supported", prop.streamPrioritiesSupported);
    setProp("global_l1_cache_supported", prop.globalL1CacheSupported);
    setProp("local_l1_cache_supported", prop.localL1CacheSupported);
    setProp("shared_mem_per_multiprocessor", prop.sharedMemPerMultiprocessor);
    setProp("regs_per_multiprocessor", prop.regsPerMultiprocessor);
    setProp("managed_memory", prop.managedMemory);
    setProp("is_multi_gpu_board", prop.isMultiGpuBoard);
    setProp("multi_gpu_board_group_id", prop.multiGpuBoardGroupID);
    setProp("host_native_atomic_supported", prop.hostNativeAtomicSupported);
    setProp("single_to_double_precision_perf_ratio", prop.singleToDoublePrecisionPerfRatio);
    setProp("pageable_memory_access", prop.pageableMemoryAccess);
    setProp("conccurrent_managed_access", prop.concurrentManagedAccess);
    setProp("compute_preemption_supported", prop.computePreemptionSupported);
    setProp(
        "can_use_host_pointer_for_registered_mem",
        prop.canUseHostPointerForRegisteredMem
    );
    setProp("cooperative_launch", prop.cooperativeLaunch);
    setProp("cooperative_multi_device_launch", prop.cooperativeMultiDeviceLaunch);
    setProp("shared_mem_per_block_optin", prop.sharedMemPerBlockOptin);
    setProp(
        "pageable_memory_access_uses_host_page_tables",
        prop.pageableMemoryAccessUsesHostPageTables
    );
    setProp("direct_managed_mem_access_from_host", prop.directManagedMemAccessFromHost);
    setProp("max_blocks_per_multi_processor", prop.maxBlocksPerMultiProcessor);
    setProp("access_policy_max_window_size", prop.accessPolicyMaxWindowSize);
    setProp("reserved_shared_mem_per_block", prop.reservedSharedMemPerBlock);
    setProp("host_register_supported", prop.hostRegisterSupported);
    setProp("sparse_hip_array_supported", prop.sparseHipArraySupported);
    setProp("host_register_read_only_supported", prop.hostRegisterReadOnlySupported);
    setProp("timeline_semaphore_interop_supported", prop.timelineSemaphoreInteropSupported);
    setProp("memory_pools_supported", prop.memoryPoolsSupported);
    setProp("gpu_direct_rdma_supported", prop.gpuDirectRDMASupported);
    setProp("gpu_direct_rdma_flush_writes_options", prop.gpuDirectRDMAFlushWritesOptions);
    setProp("gpu_direct_rdma_writes_ordering", prop.gpuDirectRDMAWritesOrdering);
    setProp("memory_pool_supported_handle_types", prop.memoryPoolSupportedHandleTypes);
    setProp("deferred_mapping_hip_array_supported", prop.deferredMappingHipArraySupported);
    setProp("ipc_event_supported", prop.ipcEventSupported);
    setProp("cluster_launch", prop.clusterLaunch);
    setProp("unified_function_pointers", prop.unifiedFunctionPointers);
    setProp("gcn_arch_name", prop.gcnArchName);
    setProp("max_shared_memory_per_multi_processor", prop.maxSharedMemoryPerMultiProcessor);
    setProp("clock_instruction_rate", prop.clockInstructionRate);
    setProp("arch", prop.arch);
    setProp("cooperative_multi_device_unmatched_func", prop.cooperativeMultiDeviceUnmatchedFunc);
    setProp("cooperative_multi_device_unmatced_grid_dim", prop.cooperativeMultiDeviceUnmatchedGridDim);
    setProp("cooperative_multi_device_unmatced_block_dim", prop.cooperativeMultiDeviceUnmatchedBlockDim);
    setProp("cooperative_multi_device_unmatced_shared_mem", prop.cooperativeMultiDeviceUnmatchedSharedMem);
    setProp("is_large_bar", prop.isLargeBar);
    setProp("asic_revision", prop.asicRevision);
};


struct TicketSemaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void init(intptr_t num) noexcept {
        current.store(num, std::memory_order::seq_cst);
    }

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order::acquire) };
            if (tk < curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};

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
            (void) deleter(x);
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
    Resource<uint8_t *, hipHostFree> h_data;
    Resource<uint8_t *, hipFree> d_data;
    size_t size;
};

struct InferenceInstance {
    MemoryResource src;
    MemoryResource dst;
    Resource<migraphx_program_parameters_t, migraphx_program_parameters_destroy> params;
    Resource<migraphx_argument_t, migraphx_argument_destroy> src_argument;
    Resource<migraphx_argument_t, migraphx_argument_destroy> dst_argument;
    Resource<hipStream_t, hipStreamDestroy> stream;
};

struct vsMIGXData {
    std::vector<VSNodeRef *> nodes;
    std::unique_ptr<VSVideoInfo> out_vi;

    std::array<int, 4> src_tile_shape, dst_tile_shape;
    int overlap_w, overlap_h;

    int device_id;

    migraphx_program_t program;
    std::vector<InferenceInstance> instances;
    std::vector<int> tickets;
    std::mutex ticket_lock;
    TicketSemaphore semaphore;

    std::string flexible_output_prop;

    [[nodiscard]] int acquire() noexcept {
        semaphore.acquire();
        {
            std::lock_guard<std::mutex> lock(ticket_lock);
            int ticket = tickets.back();
            tickets.pop_back();
            return ticket;
        }
    }

    void release(int ticket) noexcept {
        {
            std::lock_guard<std::mutex> lock(ticket_lock);
            tickets.push_back(ticket);
        }
        semaphore.release();
    }
};


static void VS_CC vsMIGXInit(
    VSMap *in,
    VSMap *out,
    void **instanceData,
    VSNode *node,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsMIGXData *>(*instanceData);
    vsapi->setVideoInfo(d->out_vi.get(), 1, node);
}


static const VSFrameRef *VS_CC vsMIGXGetFrame(
    int n,
    int activationReason,
    void **instanceData,
    void **frameData,
    VSFrameContext *frameCtx,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsMIGXData *>(*instanceData);

    if (activationReason == arInitial) {
        for (const auto & node : d->nodes) {
            vsapi->requestFrameFilter(n, node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        std::vector<const VSVideoInfo *> in_vis;
        in_vis.reserve(std::size(d->nodes));
        for (const auto & node : d->nodes) {
            in_vis.emplace_back(vsapi->getVideoInfo(node));
        }

        std::vector<const VSFrameRef *> src_frames;
        src_frames.reserve(std::size(d->nodes));
        for (const auto & node : d->nodes) {
            src_frames.emplace_back(vsapi->getFrameFilter(n, node, frameCtx));
        }

        auto src_stride = vsapi->getStride(src_frames.front(), 0);
        auto src_width = vsapi->getFrameWidth(src_frames.front(), 0);
        auto src_height = vsapi->getFrameHeight(src_frames.front(), 0);
        auto src_bytes = vsapi->getFrameFormat(src_frames.front())->bytesPerSample;

        VSFrameRef * const dst_frame = vsapi->newVideoFrame(
            d->out_vi->format, d->out_vi->width, d->out_vi->height,
            src_frames.front(), core
        );

        std::vector<VSFrameRef *> dst_frames;

        auto dst_stride = vsapi->getStride(dst_frame, 0);
        auto dst_bytes = vsapi->getFrameFormat(dst_frame)->bytesPerSample;

        auto ticket = d->acquire();
        InferenceInstance & instance = d->instances[ticket];

        auto src_tile_h = d->src_tile_shape[2];
        auto src_tile_w = d->src_tile_shape[3];
        auto src_tile_w_bytes = src_tile_w * src_bytes;
        auto src_tile_bytes = src_tile_h * src_tile_w_bytes;

        std::vector<const uint8_t *> src_ptrs;
        src_ptrs.reserve(d->src_tile_shape[1]);
        for (unsigned i = 0; i < std::size(d->nodes); ++i) {
            for (int j = 0; j < in_vis[i]->format->numPlanes; ++j) {
                src_ptrs.emplace_back(vsapi->getReadPtr(src_frames[i], j));
            }
        }

        auto step_w = src_tile_w - 2 * d->overlap_w;
        auto step_h = src_tile_h - 2 * d->overlap_h;

        auto dst_tile_h = d->dst_tile_shape[2];
        auto dst_tile_w = d->dst_tile_shape[3];
        auto dst_tile_w_bytes = dst_tile_w * dst_bytes;
        auto dst_tile_bytes = dst_tile_h * dst_tile_w_bytes;
        auto dst_planes = d->dst_tile_shape[1];

        std::vector<uint8_t *> dst_ptrs;
        if (d->flexible_output_prop.empty()) {
            for (int i = 0; i < dst_planes; ++i) {
                dst_ptrs.emplace_back(vsapi->getWritePtr(dst_frame, i));
            }
        } else {
            for (int i = 0; i < dst_planes; ++i) {
                auto frame { vsapi->newVideoFrame(
                    d->out_vi->format, d->out_vi->width, d->out_vi->height,
                    src_frames[0], core
                )};
                dst_frames.emplace_back(frame);
                dst_ptrs.emplace_back(vsapi->getWritePtr(frame, 0));
            }
        }

        auto h_scale = dst_tile_h / src_tile_h;
        auto w_scale = dst_tile_w / src_tile_w;

        const auto set_error = [&](const std::string & error_message) {
            vsapi->setFilterError(
                (__func__ + ": "s + error_message).c_str(),
                frameCtx
            );

            d->release(ticket);

            for (const auto & frame : dst_frames) {
                vsapi->freeFrame(frame);
            }

            vsapi->freeFrame(dst_frame);

            for (const auto & frame : src_frames) {
                vsapi->freeFrame(frame);
            }

            return nullptr;
        };

        checkHIPError(hipSetDevice(d->device_id));

        int y = 0;
        while (true) {
            int y_crop_start = (y == 0) ? 0 : d->overlap_h;
            int y_crop_end = (y == src_height - src_tile_h) ? 0 : d->overlap_h;

            int x = 0;
            while (true) {
                int x_crop_start = (x == 0) ? 0 : d->overlap_w;
                int x_crop_end = (x == src_width - src_tile_w) ? 0 : d->overlap_w;

                {
                    uint8_t * h_data = instance.src.h_data.data;
                    for (const uint8_t * _src_ptr : src_ptrs) {
                        const uint8_t * src_ptr { _src_ptr +
                            y * src_stride + x * vsapi->getFrameFormat(src_frames[0])->bytesPerSample
                        };

                        vs_bitblt(
                            h_data, src_tile_w_bytes,
                            src_ptr, src_stride,
                            src_tile_w_bytes, src_tile_h
                        );

                        h_data += src_tile_bytes;
                    }
                }

                checkHIPError(hipMemcpyAsync(
                    instance.src.d_data.data,
                    instance.src.h_data.data,
                    instance.src.size,
                    hipMemcpyHostToDevice,
                    instance.stream
                ));

                migraphx_arguments_t outputs;

#ifdef MIGRAPHX_VERSION_TWEAK
                checkError(migraphx_program_run_async(
                    &outputs,
                    d->program,
                    instance.params,
                    instance.stream.data,
                    "ihipStream_t"
                ));
#else // MIGRAPHX_VERSION_TWEAK
                checkHIPError(hipStreamSynchronize(instance.stream));

                checkError(migraphx_program_run(
                    &outputs,
                    d->program,
                    instance.params
                ));
#endif // MIGRAPHX_VERSION_TWEAK

                checkHIPError(hipMemcpyAsync(
                    instance.dst.h_data.data,
                    instance.dst.d_data.data,
                    instance.dst.size,
                    hipMemcpyDeviceToHost,
                    instance.stream
                ));

                checkHIPError(hipStreamSynchronize(instance.stream));

                {
                    const uint8_t * h_data = instance.dst.h_data.data;
                    auto bytes_per_sample = vsapi->getFrameFormat(dst_frame)->bytesPerSample;
                    for (int plane = 0; plane < dst_planes; ++plane) {
                        uint8_t * dst_ptr {
                            dst_ptrs[plane] +
                            h_scale * y * dst_stride + w_scale * x * dst_bytes
                        };

                        vs_bitblt(
                            dst_ptr + (y_crop_start * dst_stride + x_crop_start * bytes_per_sample),
                            dst_stride,
                            h_data + (y_crop_start * dst_tile_w_bytes + x_crop_start * bytes_per_sample),
                            dst_tile_w_bytes,
                            dst_tile_w_bytes - (x_crop_start + x_crop_end) * bytes_per_sample,
                            dst_tile_h - (y_crop_start + y_crop_end)
                        );

                        h_data += dst_tile_bytes;
                    }
                }

                if (x + src_tile_w == src_width) {
                    break;
                }

                x = std::min(x + step_w, src_width - src_tile_w);
            }

            if (y + src_tile_h == src_height) {
                break;
            }

            y = std::min(y + step_h, src_height - src_tile_h);
        }

        d->release(ticket);

        for (const auto & frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        if (!d->flexible_output_prop.empty()) {
            auto prop = vsapi->getFramePropsRW(dst_frame);

            for (int i = 0; i < dst_planes; i++) {
                auto key { d->flexible_output_prop + std::to_string(i) };
                vsapi->propSetFrame(prop, key.c_str(), dst_frames[i], paReplace);
                vsapi->freeFrame(dst_frames[i]);
            }
        }

        return dst_frame;
    }

    return nullptr;
}


static void VS_CC vsMIGXFree(
    void *instanceData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsMIGXData *>(instanceData);

    for (const auto & node : d->nodes) {
        vsapi->freeNode(node);
    }

    auto set_error = [](const std::string & error_message) {
        fprintf(stderr, "%s\n", error_message.c_str());
    };

    checkError(migraphx_program_destroy(d->program));

    delete d;
}


static void VS_CC vsMIGXCreate(
    const VSMap *in,
    VSMap *out,
    void *userData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<vsMIGXData>() };

    int num_nodes = vsapi->propNumElements(in, "clips");
    d->nodes.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        d->nodes.emplace_back(vsapi->propGetNode(in, "clips", i, nullptr));
    }

    auto set_error = [&](const std::string & error_message) {
        vsapi->setError(out, (__func__ + ": "s + error_message).c_str());
        for (const auto & node : d->nodes) {
            vsapi->freeNode(node);
        }
    };

    checkError(migraphx_program_create(&d->program));
    {
        migraphx_file_options_t file_options;
        checkError(migraphx_file_options_create(&file_options));
        const char * program_path = vsapi->propGetData(in, "program_path", 0, nullptr);
        checkError(migraphx_load(&d->program, program_path, file_options));
        checkError(migraphx_file_options_destroy(file_options));
    }

    std::vector<const VSVideoInfo *> in_vis;
    in_vis.reserve(std::size(d->nodes));
    for (const auto & node : d->nodes) {
        in_vis.emplace_back(vsapi->getVideoInfo(node));
    }

    if (auto err = checkNodes(in_vis); err.has_value()) {
        return set_error(err.value());
    }

    d->out_vi = std::make_unique<VSVideoInfo>(*in_vis.front()); // mutable


    int error;

    d->device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        d->device_id = 0;
    }

    checkHIPError(hipSetDevice(d->device_id));

    int error1, error2;
    d->overlap_w = int64ToIntS(vsapi->propGetInt(in, "overlap", 0, &error1));
    d->overlap_h = int64ToIntS(vsapi->propGetInt(in, "overlap", 1, &error2));
    if (!error1) {
        if (error2) {
            d->overlap_h = d->overlap_w;
        }

        if (d->overlap_w < 0 || d->overlap_h < 0) {
            return set_error("\"overlap\" must be non-negative");
        }
    } else {
        d->overlap_w = 0;
        d->overlap_h = 0;
    }

    size_t tile_w = static_cast<size_t>(vsapi->propGetInt(in, "tilesize", 0, &error1));
    size_t tile_h = static_cast<size_t>(vsapi->propGetInt(in, "tilesize", 1, &error2));
    if (!error1) { // manual specification triggered
        if (error2) {
            tile_h = tile_w;
        }
    } else {
        if (d->overlap_w != 0 || d->overlap_h != 0) {
            return set_error("\"tilesize\" must be specified");
        }

        // set tile size to video dimensions
        tile_w = in_vis.front()->width;
        tile_h = in_vis.front()->height;
    }
    if (tile_w - 2 * d->overlap_w <= 0 || tile_h - 2 * d->overlap_h <= 0) {
        return set_error("\"overlap\" too large");
    }

    auto flexible_output_prop = vsapi->propGetData(in, "flexible_output_prop", 0, &error);
    if (!error) {
        d->flexible_output_prop = flexible_output_prop;
    }

    const char * input_name[2];
    const_migraphx_shape_t input_shape;
    size_t input_size;
    {
        migraphx_program_parameter_shapes_t input_shapes;
        checkError(migraphx_program_get_parameter_shapes(&input_shapes, d->program));
        size_t num_inputs;
        checkError(migraphx_program_parameter_shapes_size(&num_inputs, input_shapes));
        if (num_inputs != 2) {
            return set_error("program must have exactly one input");
        }
        checkError(migraphx_program_parameter_shapes_names(&input_name[0], input_shapes));
        // here we assume that the second parameter corresponds to the input node
        checkError(migraphx_program_parameter_shapes_get(&input_shape, input_shapes, input_name[1]));
        // TODO: support dynamic shapes
#ifdef MIGRAPHX_VERSION_TWEAK
        bool is_dynamic;
        checkError(migraphx_shape_dynamic(&is_dynamic, input_shape));
        // TODO
        if (is_dynamic) {
            return set_error("dynamic shape is not supported for now");
        }
#endif // MIGRAPHX_VERSION_TWEAK
        migraphx_shape_datatype_t type;
        checkError(migraphx_shape_type(&type, input_shape));
        if (type != migraphx_shape_float_type && type != migraphx_shape_half_type) {
            return set_error("input type must be float or half");
        }
        if (in_vis[0]->format->sampleType != getSampleType(type)) {
            return set_error("sample type mismatch");
        }
        if (in_vis[0]->format->bytesPerSample != getBytesPerSample(type)) {
            return set_error("bytes per sample mismatch");
        }
        const size_t * lengths;
        size_t ndim;
        checkError(migraphx_shape_lengths(&lengths, &ndim, input_shape));
        if (ndim != 4) {
            return set_error("number of input dimension must be 4");
        }
        if (lengths[0] != 1) {
            return set_error("batch size must be 1");
        }
        if (auto num_planes = numPlanes(in_vis); static_cast<int>(lengths[1]) != num_planes) {
            return set_error("expects " + std::to_string(lengths[1]) + " input planes");
        }
        // TODO: select
        if (lengths[2] != tile_h || lengths[3] != tile_w) {
            return set_error(
                "invalid tile size, must be " +
                std::to_string(lengths[3]) + 'x' + std::to_string(lengths[2])
            );
        }
        const size_t * strides;
        checkError(migraphx_shape_strides(&strides, &ndim, input_shape));
        {
            size_t target = 1; // MIGX uses elements to measure strides
            for (int i = static_cast<int>(ndim) - 1; i >= 0; i--) {
                if (strides[i] != target) {
                    return set_error(
                        "invalid stride for NCHW, expects " +
                        std::to_string(target) +
                        " instead of " +
                        std::to_string(strides[i])
                    );
                }
                target *= lengths[i];
            }
        }
        checkError(migraphx_shape_bytes(&input_size, input_shape));
        for (int i = 0; i < 4; i++) {
            d->src_tile_shape[i] = static_cast<int>(lengths[i]);
        }
    }

    size_t output_size;
    const_migraphx_shape_t output_shape;
    int bitsPerSample;
    {
        migraphx_shapes_t output_shapes;
        checkError(migraphx_program_get_output_shapes(&output_shapes, d->program));
        size_t num_outputs;
        checkError(migraphx_shapes_size(&num_outputs, output_shapes));
        if (num_outputs != 1) {
            return set_error("program must have exactly one output");
        }
        checkError(migraphx_shapes_get(&output_shape, output_shapes, 0));
        // TODO: support dynamic shapes
#ifdef MIGRAPHX_VERSION_TWEAK
        bool is_dynamic;
        checkError(migraphx_shape_dynamic(&is_dynamic, output_shape));
        // TODO
        if (is_dynamic) {
            return set_error("dynamic shape is not supported for now");
        }
#endif // MIGRAPHX_VERSION_TWEAK
        migraphx_shape_datatype_t type;
        checkError(migraphx_shape_type(&type, output_shape));
        if (type != migraphx_shape_float_type && type != migraphx_shape_half_type) {
            return set_error("output type must be float or half");
        }
        bitsPerSample = type == migraphx_shape_float_type ? 32 : 16;
        const size_t * lengths;
        size_t ndim;
        checkError(migraphx_shape_lengths(&lengths, &ndim, output_shape));
        if (ndim != 4) {
            return set_error("number of output dimension must be 4");
        }
        if (lengths[0] != 1) {
            return set_error("batch size must be 1");
        }
        if (lengths[1] != 1 && lengths[1] != 3 && d->flexible_output_prop.empty()) {
            return set_error("output should have 1 or 3 channels, or enable \"flexible_output\"");
        }
        if (lengths[2] % tile_h != 0 && lengths[3] % tile_w != 0) {
            return set_error("output dimensions should be integer multiple of input dimensions");
        }
        const size_t * strides;
        checkError(migraphx_shape_strides(&strides, &ndim, output_shape));
        {
            size_t target = 1; // MIGX uses elements to measure strides
            for (int i = static_cast<int>(ndim) - 1; i >= 0; i--) {
                if (strides[i] != target) {
                    return set_error(
                        "invalid stride for NCHW, expects " +
                        std::to_string(target) +
                        " instead of " +
                        std::to_string(strides[i])
                    );
                }
                target *= lengths[i];
            }
        }
        checkError(migraphx_shape_bytes(&output_size, output_shape));
        for (int i = 0; i < 4; i++) {
            d->dst_tile_shape[i] = static_cast<int>(lengths[i]);
        }
    }

    int num_streams = int64ToIntS(vsapi->propGetInt(in, "num_streams", 0, &error));
    if (error) {
        num_streams = 1;
    }
    if (num_streams <= 0) {
        return set_error("\"num_streams\" must be positive");
    }

    setDimensions(
        d->out_vi,
        d->src_tile_shape,
        d->dst_tile_shape,
        bitsPerSample,
        core,
        vsapi,
        !d->flexible_output_prop.empty()
    );

    // per-stream context
    d->instances.reserve(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        InferenceInstance instance;

        checkHIPError(hipMalloc(&instance.src.d_data.data, input_size));
        checkHIPError(hipHostMalloc(
            &instance.src.h_data.data,
            input_size,
            hipHostMallocWriteCombined | hipHostMallocNonCoherent
        ));
        instance.src.size = input_size;

        checkHIPError(hipMalloc(&instance.dst.d_data.data, output_size));
        checkHIPError(hipHostMalloc(&instance.dst.h_data.data, output_size, hipHostMallocNonCoherent));
        instance.dst.size = output_size;

        checkError(migraphx_program_parameters_create(&instance.params.data));
        checkError(migraphx_argument_create(&instance.dst_argument.data, output_shape, instance.dst.d_data.data));
        checkError(migraphx_program_parameters_add(instance.params, input_name[0], instance.dst_argument));
        checkError(migraphx_argument_create(&instance.src_argument.data, input_shape, instance.src.d_data.data));
        checkError(migraphx_program_parameters_add(instance.params, input_name[1], instance.src_argument));

        checkHIPError(hipStreamCreateWithFlags(&instance.stream.data, hipStreamNonBlocking));

        d->instances.emplace_back(std::move(instance));
    }

    d->semaphore.init(num_streams);
    d->tickets.reserve(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        d->tickets.push_back(i);
    }

    if (!d->flexible_output_prop.empty()) {
        vsapi->propSetInt(out, "num_planes", d->dst_tile_shape[1], paReplace);
    }

    vsapi->createFilter(
        in, out, "Model",
        vsMIGXInit, vsMIGXGetFrame, vsMIGXFree,
        fmParallel, 0, d.release(), core
    );
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc,
    VSRegisterFunction registerFunc,
    VSPlugin *plugin
) noexcept {
    myself = plugin;

    configFunc(
        "io.github.amusementclub.vs_migraphx", "migx",
        "MIGraphX ML Filter Runtime",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    registerFunc("Model",
        "clips:clip[];"
        "program_path:data;"
        "overlap:int[]:opt;"
        "tilesize:int[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;"
        "flexible_output_prop:data:opt;"
        , vsMIGXCreate,
        nullptr,
        plugin
    );

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);

#ifndef NO_MIGX_VERSION
        vsapi->propSetData(
            out, "migraphx_version_build",
            (std::to_string(MIGRAPHX_VERSION_MAJOR) +
             "." +
             std::to_string(MIGRAPHX_VERSION_MINOR) +
             "." +
             std::to_string(MIGRAPHX_VERSION_PATCH)
            ).c_str(), -1, paReplace
        );
#endif // NO_MIGX_VERSION

        int runtime_version;
        (void) hipRuntimeGetVersion(&runtime_version);
        vsapi->propSetData(
            out, "hip_runtime_version",
            std::to_string(runtime_version).c_str(), -1, paReplace
        );

        vsapi->propSetInt(out, "hip_runtime_version_build", HIP_VERSION, paReplace);

        vsapi->propSetData(out, "path", vsapi->getPluginPath(myself), -1, paReplace);
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);

    registerFunc("DeviceProperties", "device_id:int:opt;", getDeviceProp, nullptr, plugin);
}
