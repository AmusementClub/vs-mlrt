#include <cstdio>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <cuda_runtime.h>
#include <NvInferRuntime.h>
#ifdef USE_NVINFER_PLUGIN
#include <NvInferPlugin.h>
#endif

#include "config.h"
#include "inference_helper.h"
#include "trt_utils.h"
#include "utils.h"

#ifdef _WIN32
#include <locale>
#include <codecvt>
static std::wstring translateName(const char *name) {
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	return converter.from_bytes(name);
}
#else
#define translateName(n) (n)
#endif

using namespace std::string_literals;

static const VSPlugin * myself = nullptr;

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

struct vsTrtData {
    std::vector<VSNodeRef *> nodes;
    std::unique_ptr<VSVideoInfo> out_vi;

    int device_id;
    int num_streams;
    bool use_cuda_graph;
    int overlap_w, overlap_h;

    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> engines;

    TicketSemaphore semaphore;
    std::vector<int> tickets;
    std::mutex instances_lock;
    std::vector<InferenceInstance> instances;

    [[nodiscard]]
    int acquire() noexcept {
        semaphore.acquire();
        int ticket;
        {
            std::lock_guard<std::mutex> lock { instances_lock };
            ticket = tickets.back();
            tickets.pop_back();
        }
        return ticket;
    }

    void release(int ticket) noexcept {
        {
            std::lock_guard<std::mutex> lock { instances_lock };
            tickets.push_back(ticket);
        }
        semaphore.release();
    }
};

static void VS_CC vsTrtInit(
    VSMap *in,
    VSMap *out,
    void **instanceData,
    VSNode *node,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsTrtData *>(*instanceData);
    vsapi->setVideoInfo(d->out_vi.get(), 1, node);
}

static const VSFrameRef *VS_CC vsTrtGetFrame(
    int n,
    int activationReason,
    void **instanceData,
    void **frameData,
    VSFrameContext *frameCtx,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsTrtData *>(*instanceData);

    if (activationReason == arInitial) {
        for (const auto & node : d->nodes) {
            vsapi->requestFrameFilter(n, node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const std::vector<const VSVideoInfo *> in_vis {
            getVideoInfo(vsapi, d->nodes)
        };

        const std::vector<const VSFrameRef *> src_frames {
            getFrames(n, vsapi, frameCtx, d->nodes)
        };

        const int ticket { d->acquire() };
        InferenceInstance & instance { d->instances[ticket] };

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        auto input_name = d->engines[0]->getIOTensorName(0);
        const nvinfer1::Dims src_dim { instance.exec_context->getTensorShape(input_name) };
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        const nvinfer1::Dims src_dim { instance.exec_context->getBindingDimensions(0) };
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

        const int src_planes { static_cast<int>(src_dim.d[1]) };
        const int src_tile_h { static_cast<int>(src_dim.d[2]) };
        const int src_tile_w { static_cast<int>(src_dim.d[3]) };

        std::vector<const uint8_t *> src_ptrs;
        src_ptrs.reserve(src_planes);
        for (int i = 0; i < std::ssize(d->nodes); ++i) {
            for (int j = 0; j < in_vis[i]->format->numPlanes; ++j) {
                src_ptrs.emplace_back(vsapi->getReadPtr(src_frames[i], j));
            }
        }

        VSFrameRef * const dst_frame { vsapi->newVideoFrame(
            d->out_vi->format, d->out_vi->width, d->out_vi->height,
            src_frames[0], core
        )};

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        auto output_name = d->engines[0]->getIOTensorName(1);
        const nvinfer1::Dims dst_dim { instance.exec_context->getTensorShape(output_name) };
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        const nvinfer1::Dims dst_dim { instance.exec_context->getBindingDimensions(1) };
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

        const int dst_planes { static_cast<int>(dst_dim.d[1]) };
        const int dst_tile_h { static_cast<int>(dst_dim.d[2]) };
        const int dst_tile_w { static_cast<int>(dst_dim.d[3]) };

        std::vector<uint8_t *> dst_ptrs;
        dst_ptrs.reserve(dst_planes);
        for (int i = 0; i < dst_planes; ++i) {
            dst_ptrs.emplace_back(vsapi->getWritePtr(dst_frame, i));
        }

        const int h_scale = dst_tile_h / src_tile_h;
        const int w_scale = dst_tile_w / src_tile_w;

        const IOInfo info {
            .in = InputInfo {
                .width = vsapi->getFrameWidth(src_frames[0], 0),
                .height = vsapi->getFrameHeight(src_frames[0], 0),
                .pitch = vsapi->getStride(src_frames[0], 0),
                .bytes_per_sample = vsapi->getFrameFormat(src_frames[0])->bytesPerSample,
                .tile_w = src_tile_w,
                .tile_h = src_tile_h
            },
            .out = OutputInfo {
                .pitch = vsapi->getStride(dst_frame, 0),
                .bytes_per_sample = vsapi->getFrameFormat(dst_frame)->bytesPerSample
            },
            .w_scale = w_scale,
            .h_scale = h_scale,
            .overlap_w = d->overlap_w,
            .overlap_h = d->overlap_h
        };

        const auto inference_result = inference(
            instance,
            d->device_id, d->use_cuda_graph,
            info, src_ptrs, dst_ptrs
        );

        d->release(ticket);

        for (const auto & frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        if (inference_result.has_value()) {
            vsapi->setFilterError(
                (__func__ + ": "s + inference_result.value()).c_str(),
                frameCtx
            );

            vsapi->freeFrame(dst_frame);

            return nullptr;
        }

        return dst_frame;
    }

    return nullptr;
}

static void VS_CC vsTrtFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsTrtData *>(instanceData);

    for (const auto & node : d->nodes) {
        vsapi->freeNode(node);
    }

    cudaSetDevice(d->device_id);

    delete d;
}

static void VS_CC vsTrtCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<vsTrtData>() };

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

    const char * engine_path = vsapi->propGetData(in, "engine_path", 0, nullptr);

    std::vector<const VSVideoInfo *> in_vis;
    in_vis.reserve(std::size(d->nodes));
    for (const auto & node : d->nodes) {
        in_vis.emplace_back(vsapi->getVideoInfo(node));
    }
    if (auto err = checkNodes(in_vis); err.has_value()) {
        return set_error(err.value());
    }

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

    int tile_w = int64ToIntS(vsapi->propGetInt(in, "tilesize", 0, &error1));
    int tile_h = int64ToIntS(vsapi->propGetInt(in, "tilesize", 1, &error2));

    TileSize tile_size;
    if (!error1) { // manual specification triggered
        if (error2) {
            tile_h = tile_w;
        }

        if (tile_w - 2 * d->overlap_w <= 0 || tile_h - 2 * d->overlap_h <= 0) {
            return set_error("\"overlap\" too large");
        }

        tile_size = RequestedTileSize {
            .tile_w = tile_w,
            .tile_h = tile_h
        };
    } else {
        if (d->overlap_w != 0 || d->overlap_h != 0) {
            return set_error("\"tilesize\" must be specified");
        }

        int width = in_vis[0]->width;
        int height = in_vis[0]->height;

        if (width - 2 * d->overlap_w <= 0 || height - 2 * d->overlap_h <= 0) {
            return set_error("\"overlap\" too large");
        }

        tile_size = VideoSize {
            .width = width,
            .height = height
        };
    }

    int error;

    int device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }

    int device_count;
    checkError(cudaGetDeviceCount(&device_count));
    if (0 <= device_id && device_id < device_count) {
        checkError(cudaSetDevice(device_id));
    } else {
        return set_error("invalid device ID (" + std::to_string(device_id) + ")");
    }
    d->device_id = device_id;

    d->use_cuda_graph = !!vsapi->propGetInt(in, "use_cuda_graph", 0, &error);
    if (error) {
        d->use_cuda_graph = false;
    }

    d->num_streams = int64ToIntS(vsapi->propGetInt(in, "num_streams", 0, &error));
    if (error) {
        d->num_streams = 1;
    }

    int verbosity = int64ToIntS(vsapi->propGetInt(in, "verbosity", 0, &error));
    if (error) {
        verbosity = int(nvinfer1::ILogger::Severity::kWARNING);
    }
    d->logger.set_verbosity(static_cast<nvinfer1::ILogger::Severity>(verbosity));

#ifdef USE_NVINFER_PLUGIN
    // related to https://github.com/AmusementClub/vs-mlrt/discussions/65, for unknown reason
#if !(NV_TENSORRT_MAJOR == 9 && defined(_WIN32))
    if (!initLibNvInferPlugins(&d->logger, "")) {
        vsapi->logMessage(mtWarning, "vsTrt: Initialize TensorRT plugins failed");
    }
#endif
#endif

    std::ifstream engine_stream {
        translateName(engine_path),
        std::ios::binary | std::ios::ate
    };

    if (!engine_stream.good()) {
        return set_error("open engine failed");
    }

    size_t engine_nbytes = engine_stream.tellg();
    std::unique_ptr<char [], decltype(&free)> engine_data {
        (char *) malloc(engine_nbytes), free
    };
    engine_stream.seekg(0, std::ios::beg);
    engine_stream.read(engine_data.get(), engine_nbytes);

    d->runtime.reset(nvinfer1::createInferRuntime(d->logger));
    auto maybe_engine = initEngine(engine_data.get(), engine_nbytes, d->runtime);
    if (std::holds_alternative<std::unique_ptr<nvinfer1::ICudaEngine>>(maybe_engine)) {
        d->engines.push_back(std::move(std::get<std::unique_ptr<nvinfer1::ICudaEngine>>(maybe_engine)));
    } else {
        return set_error(std::get<ErrorMessage>(maybe_engine));
    }

    auto maybe_profile_index = selectProfile(d->engines[0], tile_size);

    bool is_dynamic = false;
    d->instances.reserve(d->num_streams);
    for (int i = 0; i < d->num_streams; ++i) {
        auto maybe_instance = getInstance(
            d->engines.back(),
            maybe_profile_index,
            tile_size,
            d->use_cuda_graph,
            is_dynamic
        );

        // duplicates ICudaEngine instances
        //
        // According to
        // https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/developer-guide/index.html#perform-inference
        // each optimization profile can only have one execution context when using dynamic shapes
        if (is_dynamic && i < d->num_streams - 1) {
            auto maybe_engine = initEngine(engine_data.get(), engine_nbytes, d->runtime);
            if (std::holds_alternative<std::unique_ptr<nvinfer1::ICudaEngine>>(maybe_engine)) {
                d->engines.push_back(std::move(std::get<std::unique_ptr<nvinfer1::ICudaEngine>>(maybe_engine)));
            } else {
                return set_error(std::get<ErrorMessage>(maybe_engine));
            }
        }

        if (std::holds_alternative<InferenceInstance>(maybe_instance)) {
            auto instance = std::move(std::get<InferenceInstance>(maybe_instance));
            if (auto err = checkNodesAndContext(instance.exec_context, in_vis); err.has_value()) {
                return set_error(err.value());
            }
            d->instances.emplace_back(std::move(instance));
        } else {
            return set_error(std::get<ErrorMessage>(maybe_instance));
        }
    }

    d->semaphore.init(d->num_streams);
    d->tickets.reserve(d->num_streams);
    for (int i = 0; i < d->num_streams; ++i) {
        d->tickets.push_back(i);
    }

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    auto input_name = d->engines[0]->getIOTensorName(0);
    auto input_type = d->engines[0]->getTensorDataType(input_name);
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    auto input_type = d->engines[0]->getBindingDataType(0);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

    VSSampleType input_sample_type;
    {
        auto sample_type = getSampleType(input_type);
        if (sample_type == 0) {
            input_sample_type = stInteger;
        } else if (sample_type == 1) {
            input_sample_type = stFloat;
        } else {
            return set_error("unknown input sample type");
        }
    }
    auto input_bits_per_sample = getBytesPerSample(input_type) * 8;

    if (auto err = checkNodes(in_vis, input_sample_type, input_bits_per_sample); err.has_value()) {
        return set_error(err.value());
    }

    d->out_vi = std::make_unique<VSVideoInfo>(*in_vis[0]);

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    auto output_name = d->engines[0]->getIOTensorName(1);
    auto output_type = d->engines[0]->getTensorDataType(output_name);
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    auto output_type = d->engines[0]->getBindingDataType(1);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

    VSSampleType output_sample_type;
    {
        auto sample_type = getSampleType(output_type);
        if (sample_type == 0) {
            output_sample_type = stInteger;
        } else if (sample_type == 1) {
            output_sample_type = stFloat;
        } else {
            return set_error("unknown output sample type");
        }
    }
    auto output_bits_per_sample = getBytesPerSample(output_type) * 8;

    setDimensions(
        d->out_vi, d->instances[0].exec_context, core, vsapi,
        output_sample_type, output_bits_per_sample
    );

    vsapi->createFilter(
        in, out, "Model",
        vsTrtInit, vsTrtGetFrame, vsTrtFree,
        fmParallel, 0, d.release(), core
    );
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc,
    VSRegisterFunction registerFunc,
    VSPlugin *plugin
) noexcept {

    configFunc(
        "io.github.amusementclub.vs_tensorrt", "trt",
        "TensorRT ML Filter Runtime",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    // TRT 9 for windows does not export getInferLibVersion()
#if NV_TENSORRT_MAJOR == 9 && defined(_WIN32)
    auto test = getPluginRegistry();

    if (test == nullptr) {
        std::fprintf(stderr, "vstrt: TensorRT failed to load.\n");
        return;
    }
#else // NV_TENSORRT_MAJOR == 9 && defined(_WIN32)
    int ver = getInferLibVersion(); // must ensure this is the first nvinfer function called
#ifdef _WIN32
    if (ver == 0) { // a sentinel value, see dummy function in win32.cpp.
        std::fprintf(stderr, "vstrt: TensorRT failed to load.\n");
        return;
    }
#endif // _WIN32
    if (ver != NV_TENSORRT_VERSION) {
#if NV_TENSORRT_MAJOR >= 10
        std::fprintf(
            stderr,
            "vstrt: TensorRT version mismatch, built with %ld but loaded with %d; continue but fingers crossed...\n",
            NV_TENSORRT_VERSION,
            ver
        );
#else // NV_TENSORRT_MAJOR >= 10
        std::fprintf(
            stderr,
            "vstrt: TensorRT version mismatch, built with %d but loaded with %d; continue but fingers crossed...\n",
            NV_TENSORRT_VERSION,
            ver
        );
#endif // NV_TENSORRT_MAJOR >= 10
    }
#endif // NV_TENSORRT_MAJOR == 9 && defined(_WIN32)

    myself = plugin;

    registerFunc("Model",
        "clips:clip[];"
        "engine_path:data;"
        "overlap:int[]:opt;"
        "tilesize:int[]:opt;"
        "device_id:int:opt;"
        "use_cuda_graph:int:opt;"
        "num_streams:int:opt;"
        "verbosity:int:opt;",
        vsTrtCreate,
        nullptr,
        plugin
    );

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);

        vsapi->propSetData(
            out, "tensorrt_version",
#if NV_TENSORRT_MAJOR == 9 && defined(_WIN32)
            std::to_string(NV_TENSORRT_VERSION).c_str(), 
#else
            std::to_string(getInferLibVersion()).c_str(), 
#endif
            -1, paReplace
        );

        vsapi->propSetData(
            out, "tensorrt_version_build",
            std::to_string(NV_TENSORRT_VERSION).c_str(), -1, paReplace
        );

        int runtime_version;
        cudaRuntimeGetVersion(&runtime_version);
        vsapi->propSetData(
            out, "cuda_runtime_version",
            std::to_string(runtime_version).c_str(), -1, paReplace
        );

        vsapi->propSetData(
            out, "cuda_runtime_version_build",
            std::to_string(__CUDART_API_VERSION).c_str(), -1, paReplace
        );

        vsapi->propSetData(out, "path", vsapi->getPluginPath(myself), -1, paReplace);
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);

    registerFunc("DeviceProperties", "device_id:int:opt;", getDeviceProp, nullptr, plugin);
}
