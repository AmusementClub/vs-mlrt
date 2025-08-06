#ifndef VSTRT_UTILS_H_
#define VSTRT_UTILS_H_

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <NvInferRuntime.h>

#include <VapourSynth.h>
#include <VSHelper.h>

#ifdef __cpp_impl_reflection
#include <meta>
#endif

static inline
void setDimensions(
    std::unique_ptr<VSVideoInfo> & vi,
    const std::unique_ptr<nvinfer1::IExecutionContext> & exec_context,
    VSCore * core,
    const VSAPI * vsapi,
    int sample_type,
    int bits_per_sample,
    bool flexible_output
) noexcept {

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    auto input_name = exec_context->getEngine().getIOTensorName(0);
    auto output_name = exec_context->getEngine().getIOTensorName(1);
    const nvinfer1::Dims & in_dims = exec_context->getTensorShape(input_name);
    const nvinfer1::Dims & out_dims = exec_context->getTensorShape(output_name);
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    const nvinfer1::Dims & in_dims = exec_context->getBindingDimensions(0);
    const nvinfer1::Dims & out_dims = exec_context->getBindingDimensions(1);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

    auto in_height = static_cast<int>(in_dims.d[2]);
    auto in_width = static_cast<int>(in_dims.d[3]);

    auto out_height = static_cast<int>(out_dims.d[2]);
    auto out_width = static_cast<int>(out_dims.d[3]);

    vi->height *= out_height / in_height;
    vi->width *= out_width / in_width;

    if (out_dims.d[1] == 1 || flexible_output) {
        vi->format = vsapi->registerFormat(cmGray, sample_type, bits_per_sample, 0, 0, core);
    } else if (out_dims.d[1] == 3) {
        vi->format = vsapi->registerFormat(cmRGB, sample_type, bits_per_sample, 0, 0, core);
    }
}

static inline
std::vector<const VSVideoInfo *> getVideoInfo(
    const VSAPI * vsapi,
    const std::vector<VSNodeRef *> & nodes
) noexcept {

    std::vector<const VSVideoInfo *> vis;
    vis.reserve(std::size(nodes));

    for (const auto & node : nodes) {
        vis.emplace_back(vsapi->getVideoInfo(node));
    }

    return vis;
}

static inline
std::vector<const VSFrameRef *> getFrames(
    int n,
    const VSAPI * vsapi,
    VSFrameContext * frameCtx,
    const std::vector<VSNodeRef *> & nodes
) noexcept {

    std::vector<const VSFrameRef *> frames;
    frames.reserve(std::size(nodes));

    for (const auto & node : nodes) {
        frames.emplace_back(vsapi->getFrameFilter(n, node, frameCtx));
    }

    return frames;
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
    }

    return {};
}

static inline
std::optional<std::string> checkNodes(
    const std::vector<const VSVideoInfo *> & vis,
    int sample_type,
    int bits_per_sample
) noexcept {

    for (const auto & vi : vis) {
        if (vi->format->sampleType != sample_type) {
            return "sample type mismatch";
        }

        if (vi->format->bitsPerSample != bits_per_sample) {
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

static inline
std::optional<std::string> checkNodesAndContext(
    const std::unique_ptr<nvinfer1::IExecutionContext> & execution_context,
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    auto input_name = execution_context->getEngine().getIOTensorName(0);
    const nvinfer1::Dims & network_in_dims = execution_context->getTensorShape(input_name);
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    const nvinfer1::Dims & network_in_dims = execution_context->getBindingDimensions(0);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

    auto network_in_channels = network_in_dims.d[1];
    int num_planes = numPlanes(vis);
    if (network_in_channels != num_planes) {
        return "expects " + std::to_string(network_in_channels) + " input planes";
    }

    auto network_in_height = network_in_dims.d[2];
    auto network_in_width = network_in_dims.d[3];
    int clip_in_height = vis[0]->height;
    int clip_in_width = vis[0]->width;

    if (network_in_height > clip_in_height || network_in_width > clip_in_width) {
        return "tile size larger than clip dimension";
    }

    return {};
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

    cudaDeviceProp prop;
    if (auto error = cudaGetDeviceProperties(&prop, device_id); error != cudaSuccess) {
        vsapi->setError(out, cudaGetErrorString(error));
        return ;
    }

    auto setProp = [&](const char * name, const auto & value, int data_length = -1) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_integral_v<T>) {
            vsapi->propSetInt(out, name, static_cast<int64_t>(value), paReplace);
        } else if constexpr (std::is_same_v<T, const char *>) {
            vsapi->propSetData(out, name, value, data_length, paReplace);
        } else if constexpr (std::is_integral_v<std::remove_pointer_t<T>>) {
            std::array<int64_t, std::extent_v<std::remove_reference_t<decltype(value)>>> data;
            for (int i = 0; i < static_cast<int>(std::size(data)); i++) {
                data[i] = value[i];
            }
            vsapi->propSetIntArray(out, name, std::data(data), static_cast<int>(std::size(data)));
        }
    };

    int driver_version;
    cudaDriverGetVersion(&driver_version);
    setProp("driver_version", driver_version);

#ifdef __cpp_impl_reflection
    constexpr auto ctx = std::meta::access_context::current();
    template for (
        constexpr auto r : define_static_array(nonstatic_data_members_of(^^decltype(prop), ctx))
    ) {
        if constexpr (identifier_of(r) == "uuid") {
            std::array<int64_t, 16> uuid;
            for (int i = 0; i < 16; ++i) {
                uuid[i] = prop.uuid.bytes[i];
            }
            vsapi->propSetIntArray(out, "uuid", std::data(uuid), static_cast<int>(std::size(uuid)));
        } else if constexpr (identifier_of(r) != "reserved") {
            setProp(std::string(identifier_of(r)).c_str(), prop.[:r:]);
        }
    }
#else // __cpp_impl_reflection
    setProp("name", prop.name);
    {
        std::array<int64_t, 16> uuid;
        for (int i = 0; i < 16; ++i) {
            uuid[i] = prop.uuid.bytes[i];
        }
        vsapi->propSetIntArray(out, "uuid", std::data(uuid), static_cast<int>(std::size(uuid)));
    }
    setProp("total_global_memory", prop.totalGlobalMem);
    setProp("shared_memory_per_block", prop.sharedMemPerBlock);
    setProp("regs_per_block", prop.regsPerBlock);
    setProp("warp_size", prop.warpSize);
    setProp("mem_pitch", prop.memPitch);
    setProp("max_threads_per_block", prop.maxThreadsPerBlock);
    setProp("total_const_mem", prop.totalConstMem);
    setProp("major", prop.major);
    setProp("minor", prop.minor);
    setProp("texture_alignment", prop.textureAlignment);
    setProp("texture_pitch_alignment", prop.texturePitchAlignment);
    setProp("multi_processor_count", prop.multiProcessorCount);
    setProp("integrated", prop.integrated);
    setProp("can_map_host_memory", prop.canMapHostMemory);
    setProp("concurrent_kernels", prop.concurrentKernels);
    setProp("ecc_enabled", prop.ECCEnabled);
    setProp("pci_bus_id", prop.pciBusID);
    setProp("pci_device_id", prop.pciDeviceID);
    setProp("pci_domain_id", prop.pciDomainID);
    setProp("tcc_driver", prop.tccDriver);
    setProp("async_engine_count", prop.asyncEngineCount);
    setProp("unified_addressing", prop.unifiedAddressing);
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
    setProp("pageable_memory_access", prop.pageableMemoryAccess);
    setProp("conccurrent_managed_access", prop.concurrentManagedAccess);
    setProp("compute_preemption_supported", prop.computePreemptionSupported);
    setProp(
        "can_use_host_pointer_for_registered_mem",
        prop.canUseHostPointerForRegisteredMem
    );
    setProp("cooperative_launch", prop.cooperativeLaunch);
    setProp("shared_mem_per_block_optin", prop.sharedMemPerBlockOptin);
    setProp(
        "pageable_memory_access_uses_host_page_tables",
        prop.pageableMemoryAccessUsesHostPageTables
    );
    setProp("direct_managed_mem_access_from_host", prop.directManagedMemAccessFromHost);
    setProp("max_blocks_per_multi_processor", prop.maxBlocksPerMultiProcessor);
    setProp("access_policy_max_window_size", prop.accessPolicyMaxWindowSize);
    setProp("reserved_shared_mem_per_block", prop.reservedSharedMemPerBlock);
#endif // __cpp_impl_reflection
};

#endif // VSTRT_UTILS_H_
