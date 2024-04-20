#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <variant>
#include <vector>

#if not __cpp_lib_atomic_wait
#include <chrono>
#include <thread>
using namespace std::chrono_literals;
#endif

#include <VapourSynth.h>
#include <VSHelper.h>

#include <onnx/common/version.h>
#include <onnx/onnx_pb.h>

#define NOMINMAX

#include <onnxruntime_c_api.h>
#include <onnxruntime_run_options_config_keys.h>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif // ENABLE_CUDA

#ifdef ENABLE_DML
#include <dml_provider_factory.h>
#endif // ENABLE_DML

#include "../common/convert_float_to_float16.h"
#include "../common/onnx_utils.h"

#include "config.h"


#ifdef ENABLE_COREML
extern "C" OrtStatusPtr OrtSessionOptionsAppendExecutionProvider_CoreML(OrtSessionOptions *so, int flags);
#endif // ENABLE_COREML

#define checkError(expr) do {                                                  \
    OrtStatusPtr __err = expr;                                                 \
    if (__err) {                                                               \
        const std::string message = ortapi->GetErrorMessage(__err);            \
        ortapi->ReleaseStatus(__err);                                          \
        return set_error("'"s + # expr + "' failed: " + message);              \
    }                                                                          \
} while(0)

#ifdef ENABLE_CUDA
#define checkCUDAError(expr) do {                                              \
    if (cudaError_t result = expr; result != cudaSuccess) {                    \
        const char * error_str = cudaGetErrorString(result);                   \
        return set_error("'"s + # expr + "' failed: " + error_str);            \
    }                                                                          \
} while(0)
#endif // ENABLE_CUDA

using namespace std::string_literals;

static const VSPlugin * myself = nullptr;
static const OrtApi * ortapi = nullptr;
static std::atomic<int64_t> logger_id = 0;
static std::mutex capture_lock;


// rename GridSample to com.microsoft::GridSample
// onnxruntime has support for CUDA-accelerated GridSample only in its own opset domain
static void rename(ONNX_NAMESPACE::ModelProto & model) {
#if ORT_API_VERSION < 18
    constexpr auto ms_domain = "com.microsoft";

    bool has_ms_opset = false;
    for (const auto & opset : model.opset_import()) {
        if (opset.has_domain() && opset.domain() == ms_domain) {
            has_ms_opset = true;
            break;
        }
    }

    if (!has_ms_opset) {
        ONNX_NAMESPACE::OperatorSetIdProto opset_id;
        *opset_id.mutable_domain() = ms_domain;
        opset_id.set_version(1);
        *model.add_opset_import() = std::move(opset_id);
    }

    for (auto & node : *model.mutable_graph()->mutable_node()) {
        if (node.has_op_type() && node.op_type() == "GridSample") {
            *node.mutable_domain() = ms_domain;
        }
    }
#endif // ORT_API_VERSION < 18
}


[[nodiscard]]
static std::optional<std::string> ortInit() noexcept {
    static std::once_flag ort_init_flag;

    std::call_once(ort_init_flag, []() {
        auto p = OrtGetApiBase();
        if (p)
            ortapi = p->GetApi(ORT_API_VERSION);
    });

    if (ortapi) {
        return {};
    } else {
        return "ONNX Runtime initialization failed";
    }
}


[[nodiscard]]
static std::variant<std::string, std::array<int64_t, 4>> getShape(
    const OrtTensorTypeAndShapeInfo* tensor_info
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    std::array<int64_t, 4> shape;
    checkError(ortapi->GetDimensions(tensor_info, std::data(shape), std::size(shape)));

    return shape;
}


[[nodiscard]]
static std::variant<std::string, std::array<int64_t, 4>> getShape(
    const OrtSession * session,
    bool input
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    OrtTypeInfo * typeinfo;
    if (input) {
        checkError(ortapi->SessionGetInputTypeInfo(session, 0, &typeinfo));
    } else {
        checkError(ortapi->SessionGetOutputTypeInfo(session, 0, &typeinfo));
    }

    const OrtTensorTypeAndShapeInfo* tensor_info;
    checkError(ortapi->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    auto maybe_shape = getShape(tensor_info);
    ortapi->ReleaseTypeInfo(typeinfo);

    if (std::holds_alternative<std::string>(maybe_shape)) {
        return set_error(std::get<std::string>(maybe_shape));
    }

    return std::get<std::array<int64_t, 4>>(maybe_shape);
}

static size_t getNumBytes(int32_t type) {
    using namespace ONNX_NAMESPACE;

    switch (type) {
        case TensorProto::FLOAT:
            return 4;
        case TensorProto::FLOAT16:
            return 2;
        default:
            return 0;
    }
}


static int numPlanes(
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    int num_planes = 0;

    for (const auto & vi : vis) {
        num_planes += vi->format->numPlanes;
    }

    return num_planes;
}


[[nodiscard]]
static std::optional<std::string> checkNodes(
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    for (const auto & vi : vis) {
        if (vi->format->sampleType != stFloat) {
            return "expects clip with floating-point type";
        }
        
        if (vi->format->bitsPerSample != 32 && vi->format->bitsPerSample != 16) {
            return "expects clip with type fp32 or fp16";
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


[[nodiscard]]
static std::optional<std::string> checkIOInfo(
    const OrtTypeInfo * info,
    bool is_output
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    const OrtTensorTypeAndShapeInfo * tensor_info;
    checkError(ortapi->CastTypeInfoToTensorInfo(info, &tensor_info));

    ONNXTensorElementDataType element_type;
    checkError(ortapi->GetTensorElementType(tensor_info, &element_type));

    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        return set_error("expects network IO with type fp32 or fp16");
    }

    size_t num_dims;
    checkError(ortapi->GetDimensionsCount(tensor_info, &num_dims));
    if (num_dims != 4) {
        return set_error("expects network with 4-D IO");
    }

    auto maybe_shape = getShape(tensor_info);
    if (std::holds_alternative<std::string>(maybe_shape)) {
        return set_error(std::get<std::string>(maybe_shape));
    }

    auto shape = std::get<std::array<int64_t, 4>>(maybe_shape);
    if (shape[0] != 1) {
        return set_error("batch size of network must be 1");
    }

    if (is_output) {
        int64_t out_channels = shape[1];
        if (out_channels != 1 && out_channels != 3) {
            return "output dimensions must be 1 or 3";
        }
    }

    return {};
}


[[nodiscard]]
static std::optional<std::string> checkSession(
    const OrtSession * session
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    size_t num_inputs;
    checkError(ortapi->SessionGetInputCount(session, &num_inputs));

    if (num_inputs != 1) {
        return set_error("network input count must be 1, got " + std::to_string(num_inputs));
    }

    OrtTypeInfo * input_type_info;
    checkError(ortapi->SessionGetInputTypeInfo(session, 0, &input_type_info));

    if (auto err = checkIOInfo(input_type_info, false); err.has_value()) {
        return set_error(err.value());
    }

    ortapi->ReleaseTypeInfo(input_type_info);

    size_t num_outputs;
    checkError(ortapi->SessionGetOutputCount(session, &num_outputs));

    if (num_outputs != 1) {
        return "network output count must be 1, got " + std::to_string(num_outputs);
    }

    OrtTypeInfo * output_type_info;
    checkError(ortapi->SessionGetOutputTypeInfo(session, 0, &output_type_info));

    if (auto err = checkIOInfo(output_type_info, true); err.has_value()) {
        return set_error(err.value());
    }

    ortapi->ReleaseTypeInfo(output_type_info);

    return {};
}

[[nodiscard]]
static std::optional<std::string> checkNodesAndNetwork(
    const OrtSession * session,
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    OrtTypeInfo * input_type_info;
    checkError(ortapi->SessionGetInputTypeInfo(session, 0, &input_type_info));

    const OrtTensorTypeAndShapeInfo * input_tensor_info;
    checkError(ortapi->CastTypeInfoToTensorInfo(input_type_info, &input_tensor_info));

    auto network_in_dims = std::get<std::array<int64_t, 4>>(getShape(input_tensor_info));

    int network_in_channels = static_cast<int>(network_in_dims[1]);
    int num_planes = numPlanes(vis);
    if (network_in_channels != num_planes) {
        return set_error("expects " + std::to_string(network_in_channels) + " input planes");
    }

    auto network_in_height = network_in_dims[2];
    auto network_in_width = network_in_dims[3];
    auto clip_in_height = vis.front()->height;
    auto clip_in_width = vis.front()->width;
    if (network_in_height > clip_in_height || network_in_width > clip_in_width) {
        return set_error("tile size larger than clip dimension");
    }

    ortapi->ReleaseTypeInfo(input_type_info);

    return {};
}

static void setDimensions(
    std::unique_ptr<VSVideoInfo> & vi,
    const std::array<int64_t, 4> & input_shape,
    const std::array<int64_t, 4> & output_shape,
    VSCore * core,
    const VSAPI * vsapi,
    int32_t onnx_output_type
) noexcept {

    vi->height *= output_shape[2] / input_shape[2];
    vi->width *= output_shape[3] / input_shape[3];

    if (output_shape[1] == 1) {
        vi->format = vsapi->registerFormat(cmGray, stFloat, 8 * getNumBytes(onnx_output_type), 0, 0, core);
    } else if (output_shape[1] == 3) {
        vi->format = vsapi->registerFormat(cmRGB, stFloat, 8 * getNumBytes(onnx_output_type), 0, 0, core);
    }
}

struct TicketSemaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order_acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order_acquire) };
            if (tk <= curr) {
                return;
            }
#if __cpp_lib_atomic_wait
            current.wait(curr, std::memory_order::relaxed);
#else // __cpp_lib_atomic_wait
            std::this_thread::sleep_for(10ms);
#endif // __cpp_lib_atomic_wait
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order_release);
#if __cpp_lib_atomic_wait
        current.notify_all();
#endif // __cpp_lib_atomic_wait
    }
};

enum class Backend {
    CPU = 0,
    CUDA = 1,
    COREML = 2,
    DML = 3
};

#ifdef ENABLE_CUDA
struct CUDA_Resource_t {
    uint8_t * h_data;
    uint8_t * d_data;
    size_t size;
};
#endif // ENABLE_CUDA

// per-stream context
struct Resource {
    OrtSession * session;
    OrtValue * input_tensor;
    OrtValue * output_tensor;
    OrtIoBinding * binding;
    char * input_name;
    char * output_name;

#ifdef ENABLE_CUDA
    cudaStream_t stream;
    CUDA_Resource_t input;
    CUDA_Resource_t output;
    bool require_replay;
#endif // ENABLE_CUDA
};

struct vsOrtData {
    std::vector<VSNodeRef *> nodes;
    std::unique_ptr<VSVideoInfo> out_vi;

    int overlap_w, overlap_h;

    OrtEnv * environment;
    Backend backend;

    int device_id;

    std::vector<Resource> resources;
    std::vector<int> tickets;
    std::mutex ticket_lock;
    TicketSemaphore semaphore;

    int acquire() noexcept {
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


static void VS_CC vsOrtInit(
    VSMap *in,
    VSMap *out,
    void **instanceData,
    VSNode *node,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsOrtData *>(*instanceData);
    vsapi->setVideoInfo(d->out_vi.get(), 1, node);
}


static const VSFrameRef *VS_CC vsOrtGetFrame(
    int n,
    int activationReason,
    void **instanceData,
    void **frameData,
    VSFrameContext *frameCtx,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsOrtData *>(*instanceData);

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
        auto dst_stride = vsapi->getStride(dst_frame, 0);
        auto dst_bytes = vsapi->getFrameFormat(dst_frame)->bytesPerSample;

        auto ticket = d->acquire();
        Resource & resource = d->resources[ticket];

        auto src_tile_shape = std::get<std::array<int64_t, 4>>(getShape(resource.session, true));
        auto src_tile_h = src_tile_shape[2];
        auto src_tile_w = src_tile_shape[3];
        auto src_tile_w_bytes = src_tile_w * src_bytes;
        auto src_tile_bytes = src_tile_h * src_tile_w_bytes;

        std::vector<const uint8_t *> src_ptrs;
        src_ptrs.reserve(src_tile_shape[1]);
        for (unsigned i = 0; i < std::size(d->nodes); ++i) {
            for (int j = 0; j < in_vis[i]->format->numPlanes; ++j) {
                src_ptrs.emplace_back(vsapi->getReadPtr(src_frames[i], j));
            }
        }

        auto step_w = src_tile_w - 2 * d->overlap_w;
        auto step_h = src_tile_h - 2 * d->overlap_h;

        auto dst_tile_shape = std::get<std::array<int64_t, 4>>(getShape(resource.session, false));
        auto dst_tile_h = dst_tile_shape[2];
        auto dst_tile_w = dst_tile_shape[3];
        auto dst_tile_w_bytes = dst_tile_w * dst_bytes;
        auto dst_tile_bytes = dst_tile_h * dst_tile_w_bytes;
        auto dst_planes = dst_tile_shape[1];
        uint8_t * dst_ptrs[3] {};
        for (int i = 0; i < dst_planes; ++i) {
            dst_ptrs[i] = vsapi->getWritePtr(dst_frame, i);
        }

        auto h_scale = dst_tile_h / src_tile_h;
        auto w_scale = dst_tile_w / src_tile_w;

        const auto set_error = [&](const std::string & error_message) {
            vsapi->setFilterError(
                (__func__ + ": "s + error_message).c_str(),
                frameCtx
            );

            d->release(ticket);

            vsapi->freeFrame(dst_frame);

            for (const auto & frame : src_frames) {
                vsapi->freeFrame(frame);
            }

            return nullptr;
        };

        OrtRunOptions * run_options {};

#ifdef ENABLE_CUDA
        if (d->backend == Backend::CUDA) {
            checkCUDAError(cudaSetDevice(d->device_id));

#if ORT_API_VERSION >= 16
            checkError(ortapi->CreateRunOptions(&run_options));
            if (run_options == nullptr) {
                return set_error("create run_options failed");
            }
            checkError(ortapi->AddRunConfigEntry(
                run_options,
                kOrtRunOptionsConfigDisableSynchronizeExecutionProviders,
                "1"
            ));
#endif // ORT_API_VERSION >= 16
        }
#endif // ENABLE_CUDA

        int y = 0;
        while (true) {
            int y_crop_start = (y == 0) ? 0 : d->overlap_h;
            int y_crop_end = (y == src_height - src_tile_h) ? 0 : d->overlap_h;

            int x = 0;
            while (true) {
                int x_crop_start = (x == 0) ? 0 : d->overlap_w;
                int x_crop_end = (x == src_width - src_tile_w) ? 0 : d->overlap_w;

                {
                    uint8_t * input_buffer;
#ifdef ENABLE_CUDA
                    uint8_t * h_input_buffer = resource.input.h_data;
#endif // ENABLE_CUDA
                    checkError(ortapi->GetTensorMutableData(
                        resource.input_tensor,
                        reinterpret_cast<void **>(&input_buffer)
                    ));

                    for (const auto & _src_ptr : src_ptrs) {
                        const uint8_t * src_ptr { _src_ptr +
                            y * src_stride + x * src_bytes
                        };

#ifdef ENABLE_CUDA
                        if (d->backend == Backend::CUDA) {
                            vs_bitblt(
                                h_input_buffer, src_tile_w_bytes,
                                src_ptr, src_stride,
                                src_tile_w_bytes, src_tile_h
                            );
                            h_input_buffer += src_tile_bytes;
                        } else
#endif // ENABLE_CUDA
                        {
                            vs_bitblt(
                                input_buffer, src_tile_w_bytes,
                                src_ptr, src_stride,
                                src_tile_w_bytes, src_tile_h
                            );
                            input_buffer += src_tile_bytes;
                        }
                    }
                }

#ifdef ENABLE_CUDA
                if (d->backend == Backend::CUDA) {
                    checkCUDAError(cudaMemcpyAsync(
                        resource.input.d_data,
                        resource.input.h_data,
                        resource.input.size,
                        cudaMemcpyHostToDevice,
                        resource.stream
                    ));

#if ORT_API_VERSION < 16
                    checkCUDAError(cudaStreamSynchronize(resource.stream));
#endif // ORT_API_VERSION < 16
                }
#endif // ENABLE_CUDA

#ifdef ENABLE_CUDA
                if (resource.require_replay) [[unlikely]] {
                    resource.require_replay = false;

                    // runs it under a global lock
                    // onnxruntime uses global-mode stream capture on a private stream
                    // this lock prevents concurrent capture sequences in other threads
                    //
                    // note that this applies only to stream capture from the ort library
                    // this fails when another plugin also uses global-mode stream capture
                    std::lock_guard _ { capture_lock };
                    checkError(ortapi->RunWithBinding(resource.session, run_options, resource.binding));

                    // onnxruntime replays the graph itself in CUDAExecutionProvider::OnRunEnd
                } else
#endif // ENABLE_CUDA
                if (d->backend == Backend::CPU || d->backend == Backend::CUDA) {
                    checkError(ortapi->RunWithBinding(resource.session, run_options, resource.binding));
                } else {
                    checkError(ortapi->Run(
                        resource.session,
                        run_options,
                        &resource.input_name,
                        &resource.input_tensor,
                        1,
                        &resource.output_name,
                        1,
                        &resource.output_tensor
                    ));
                }

#ifdef ENABLE_CUDA
                if (d->backend == Backend::CUDA) {
                    checkCUDAError(cudaMemcpyAsync(
                        resource.output.h_data,
                        resource.output.d_data,
                        resource.output.size,
                        cudaMemcpyDeviceToHost,
                        resource.stream
                    ));
                    checkCUDAError(cudaStreamSynchronize(resource.stream));
                }
#endif // ENABLE_CUDA

                {
                    uint8_t * output_buffer;
#ifdef ENABLE_CUDA
                    uint8_t * h_output_buffer = resource.output.h_data;
#endif // ENABLE_CUDA
                    checkError(ortapi->GetTensorMutableData(
                        resource.output_tensor,
                        reinterpret_cast<void **>(&output_buffer)
                    ));

                    for (int plane = 0; plane < dst_planes; ++plane) {
                        auto dst_ptr = (dst_ptrs[plane] +
                            h_scale * y * dst_stride + w_scale * x * dst_bytes
                        );

#ifdef ENABLE_CUDA
                        if (d->backend == Backend::CUDA) {
                            vs_bitblt(
                                dst_ptr + (y_crop_start * dst_stride + x_crop_start * dst_bytes),
                                dst_stride,
                                h_output_buffer + (y_crop_start * dst_tile_w_bytes + x_crop_start * dst_bytes),
                                dst_tile_w_bytes,
                                dst_tile_w_bytes - (x_crop_start + x_crop_end) * dst_bytes,
                                dst_tile_h - (y_crop_start + y_crop_end)
                            );

                            h_output_buffer += dst_tile_bytes;
                        } else
#endif // ENABLE_CUDA
                        {
                            vs_bitblt(
                                dst_ptr + (y_crop_start * dst_stride + x_crop_start * dst_bytes),
                                dst_stride,
                                output_buffer + (y_crop_start * dst_tile_w_bytes + x_crop_start * dst_bytes),
                                dst_tile_w_bytes,
                                dst_tile_w_bytes - (x_crop_start + x_crop_end) * dst_bytes,
                                dst_tile_h - (y_crop_start + y_crop_end)
                            );

                            output_buffer += dst_tile_bytes;
                        }
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

        if (run_options) {
            ortapi->ReleaseRunOptions(run_options);
        }

        d->release(ticket);

        for (const auto & frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        return dst_frame;
    }

    return nullptr;
}


static void VS_CC vsOrtFree(
    void *instanceData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsOrtData *>(instanceData);

    for (const auto & node : d->nodes) {
        vsapi->freeNode(node);
    }

    for (const auto & resource : d->resources) {
        ortapi->ReleaseIoBinding(resource.binding);
        ortapi->ReleaseValue(resource.output_tensor);
        ortapi->ReleaseValue(resource.input_tensor);
        ortapi->ReleaseSession(resource.session);

#ifdef ENABLE_CUDA
        if (d->backend == Backend::CUDA) {
            cudaStreamDestroy(resource.stream);
            cudaFreeHost(resource.input.h_data);
            cudaFree(resource.input.d_data);
            cudaFreeHost(resource.output.h_data);
            cudaFree(resource.output.d_data);
        }
#endif // ENABLE_CUDA
    }

    ortapi->ReleaseEnv(d->environment);

    delete d;
}


static void VS_CC vsOrtCreate(
    const VSMap *in,
    VSMap *out,
    void *userData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<vsOrtData>() };

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

    auto verbosity = static_cast<OrtLoggingLevel>(
        vsapi->propGetInt(in, "verbosity", 0, &error)
    );
    if (error) {
        verbosity = ORT_LOGGING_LEVEL_WARNING;
    }

    // match verbosity of vs-trt
    verbosity = static_cast<OrtLoggingLevel>(4 - static_cast<int>(verbosity));

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

    const char * provider = vsapi->propGetData(in, "provider", 0, &error);
    if (error) {
        provider = "";
    }

    if (strlen(provider) == 0 || strcmp(provider, "CPU") == 0) {
        d->backend = Backend::CPU;
#ifdef ENABLE_CUDA
    } else if (strcmp(provider, "CUDA") == 0) {
        checkCUDAError(cudaSetDevice(d->device_id));
        d->backend = Backend::CUDA;
#endif // ENABLE_CUDA
#ifdef ENABLE_COREML
    } else if (strcmp(provider, "COREML") == 0) {
        d->backend = Backend::COREML;
#endif // ENABLE_COREML
#ifdef ENABLE_DML
    } else if (strcmp(provider, "DML") == 0) {
        d->backend = Backend::DML;
#endif // ENABLE_DML
    } else {
        return set_error("unknwon provider "s + provider);
    }

    int num_streams = int64ToIntS(vsapi->propGetInt(in, "num_streams", 0, &error));
    if (error) {
        num_streams = 1;
    }
    if (num_streams <= 0) {
        return set_error("\"num_streams\" must be positive");
    }

#ifdef ENABLE_CUDA
    bool cudnn_benchmark = !!(vsapi->propGetInt(in, "cudnn_benchmark", 0, &error));
    if (error) {
        cudnn_benchmark = true;
    }

#if ORT_API_VERSION >= 17
    bool prefer_nhwc = !!(vsapi->propGetInt(in, "prefer_nhwc", 0, &error));
    if (error) {
        prefer_nhwc = false;
    }
#endif // ORT_API_VERSION >= 17

    bool tf32 = !!(vsapi->propGetInt(in, "tf32", 0, &error));
    if (error) {
        tf32 = false;
    }
#endif // ENABLE_CUDA

    if (auto err = ortInit(); err.has_value()) {
        return set_error(err.value());
    }

    bool fp16 = !!vsapi->propGetInt(in, "fp16", 0, &error);
    if (error) {
        fp16 = false;
    }

    bool path_is_serialization = !!vsapi->propGetInt(in, "path_is_serialization", 0, &error);
    if (error) {
        path_is_serialization = false;
    }

    bool use_cuda_graph = !!vsapi->propGetInt(in, "use_cuda_graph", 0, &error);
    if (error) {
        use_cuda_graph = false;
    }

    int output_format = int64ToIntS(vsapi->propGetInt(in, "output_format", 0, &error));
    if (error) {
        output_format = 0;
    }
    if (output_format != 0 && output_format != 1) {
        return set_error("\"output_format\" must be 0 or 1");
    }

    std::string_view path_view;
    std::string path;
    if (path_is_serialization) {
        path_view = {
            vsapi->propGetData(in, "network_path", 0, nullptr),
            static_cast<size_t>(vsapi->propGetDataSize(in, "network_path", 0, nullptr))
        };
    } else {
        path = vsapi->propGetData(in, "network_path", 0, nullptr);
        bool builtin = !!vsapi->propGetInt(in, "builtin", 0, &error);
        if (builtin) {
            const char *modeldir = vsapi->propGetData(in, "builtindir", 0, &error);
            if (!modeldir) modeldir = "models";
            path = std::string(modeldir) + "/" + path;
            std::string dir { vsapi->getPluginPath(myself) };
            dir = dir.substr(0, dir.rfind('/') + 1);
            path = dir + path;
        }
        path_view = path;
    }

    auto result = loadONNX(path_view, tile_w, tile_h, path_is_serialization);
    if (std::holds_alternative<std::string>(result)) {
        return set_error(std::get<std::string>(result));
    }

    auto onnx_model = std::move(std::get<ONNX_NAMESPACE::ModelProto>(result));

    if (fp16) {
        std::unordered_set<std::string> fp16_blacklist_ops;
        int num = vsapi->propNumElements(in, "fp16_blacklist_ops");
        if (num == -1) {
            fp16_blacklist_ops = {
                "ArrayFeatureExtractor", "Binarizer", "CastMap", "CategoryMapper",
                "DictVectorizer", "FeatureVectorizer", "Imputer", "LabelEncoder",
                "LinearClassifier", "LinearRegressor", "Normalizer", "OneHotEncoder",
                "SVMClassifier", "SVMRegressor", "Scaler", "TreeEnsembleClassifier",
                "TreeEnsembleRegressor", "ZipMap", "NonMaxSuppression", "TopK",
                "RoiAlign", "Range", "CumSum", "Min", "Max", "Resize", "Upsample",
                "ReduceMean", // for CUGAN-pro
                "GridSample", // for RIFE, etc
            };
        } else {
            for (int i = 0; i < num; i++) {
                fp16_blacklist_ops.emplace(vsapi->propGetData(in, "fp16_blacklist_ops", i, nullptr));
            }
        }
        convert_float_to_float16(
            onnx_model,
            false,
            fp16_blacklist_ops,
            in_vis.front()->format->bytesPerSample == 4,
            output_format == 0
        );
    }

    rename(onnx_model);

    auto onnx_input_type = onnx_model.graph().input()[0].type().tensor_type().elem_type();
    auto onnx_output_type = onnx_model.graph().output()[0].type().tensor_type().elem_type();

    if (onnx_input_type == ONNX_NAMESPACE::TensorProto::FLOAT && in_vis.front()->format->bitsPerSample != 32) {
        return set_error("the onnx requires input to be of type fp32");
    } else if (onnx_input_type == ONNX_NAMESPACE::TensorProto::FLOAT16 && in_vis.front()->format->bitsPerSample != 16) {
        return set_error("the onnx requires input to be of type fp16");
    }

    std::string onnx_data = onnx_model.SerializeAsString();
    if (std::size(onnx_data) == 0) {
        return set_error("proto serialization failed");
    }

    // onnxruntime related code

    // environment per filter instance
    auto logger_id_str = "vs-ort" + std::to_string(logger_id.fetch_add(1, std::memory_order_relaxed));
    checkError(ortapi->CreateEnv(verbosity, logger_id_str.c_str(), &d->environment));

    OrtMemoryInfo * memory_info;
#ifdef ENABLE_CUDA
    if (d->backend == Backend::CUDA) {
        checkError(ortapi->CreateMemoryInfo(
            "Cuda", OrtDeviceAllocator, d->device_id,
            OrtMemTypeDefault, &memory_info
        ));
    } else
#endif // ENABLE_CUDA
    {
        checkError(ortapi->CreateMemoryInfo(
            "Cpu", OrtDeviceAllocator, /* device_id */ 0,
            OrtMemTypeDefault, &memory_info
        ));
    }

    OrtAllocator * cpu_allocator;
    checkError(ortapi->GetAllocatorWithDefaultOptions(&cpu_allocator));

    // per-stream context
    d->semaphore.current.store(num_streams - 1, std::memory_order_relaxed);
    d->tickets.reserve(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        d->tickets.push_back(i);
    }
    d->resources.reserve(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        Resource resource;

        OrtSessionOptions * session_options;
        checkError(ortapi->CreateSessionOptions(&session_options));
        checkError(ortapi->SetSessionExecutionMode(
            session_options,
            ExecutionMode::ORT_SEQUENTIAL
        ));

        // it is important to disable the memory pattern optimization
        // for use in vapoursynth
        //
        // this optimization merges memory allocation calls, but it is useless
        // during inference in vs since the memory usage is fixed
        //
        // it also prevents the use of cuda graphs which requires a static
        // memory configuration
        checkError(ortapi->DisableMemPattern(session_options));

        // TODO: other providers
#ifdef ENABLE_CUDA
        if (d->backend == Backend::CUDA) {
            checkCUDAError(cudaStreamCreateWithFlags(&resource.stream, cudaStreamNonBlocking));

            OrtCUDAProviderOptionsV2 * cuda_options;
            checkError(ortapi->CreateCUDAProviderOptions(&cuda_options));
#ifdef _MSC_VER
            // Preload cuda dll from vsort directory.
            static std::once_flag cuda_dll_preloaded_flag;
            static bool cuda_dll_preload_ok;
            std::call_once(cuda_dll_preloaded_flag, []() {
                    extern bool preloadCudaDlls();
                    cuda_dll_preload_ok = preloadCudaDlls();
            });
            if (!cuda_dll_preload_ok)
                return set_error("cuda DLL preloading failed");

#endif // _MSC_VER
            // should not set 'do_copy_in_default_stream' to false
            const char * keys [] {
                "device_id",
                "cudnn_conv_algo_search",
                "cudnn_conv_use_max_workspace",
                "arena_extend_strategy",
                "enable_cuda_graph",
#if ORT_API_VERSION >= 17
                "prefer_nhwc",
                "use_tf32",
#endif // ORT_API_VERSION >= 17
            };
            auto device_id_str = std::to_string(d->device_id);
            const char * values [] {
                device_id_str.c_str(),
                "EXHAUSTIVE",
                "1",
                "kSameAsRequested",
                "0",
#if ORT_API_VERSION >= 17
                "0",
                "0",
#endif // ORT_API_VERSION >= 17
            };
            if (!cudnn_benchmark) {
                values[1] = "HEURISTIC";
            }
            if (use_cuda_graph) {
                values[4] = "1";
                resource.require_replay = true;
            } else {
                resource.require_replay = false;
            }
#if ORT_API_VERSION >= 17
            if (prefer_nhwc) {
                values[5] = "1";
            }
            if (tf32) {
                values[6] = "1";
            }
#endif // ORT_API_VERSION >= 17
            checkError(ortapi->UpdateCUDAProviderOptions(cuda_options, keys, values, std::size(keys)));

#if ORT_API_VERSION >= 16
            checkError(ortapi->UpdateCUDAProviderOptionsWithValue(
                cuda_options,
                "user_compute_stream",
                resource.stream
            ));
#endif // ORT_API_VERSION >= 16

            checkError(ortapi->SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options));

            ortapi->ReleaseCUDAProviderOptions(cuda_options);
        }
#endif // ENABLE_CUDA
#ifdef ENABLE_COREML
        else if (d->backend == Backend::COREML) {
            checkError(OrtSessionOptionsAppendExecutionProvider_CoreML(
                session_options,
                0
            ));
        }
#endif // ENABLE_COREML
#ifdef ENABLE_DML
        else if (d->backend == Backend::DML) {
            const OrtDmlApi * ortdmlapi {};
            checkError(ortapi->GetExecutionProviderApi("DML", ORT_API_VERSION, (const void **) &ortdmlapi));
            checkError(ortdmlapi->SessionOptionsAppendExecutionProvider_DML(session_options, d->device_id));
        }
#endif // ENABLE_DML

        checkError(ortapi->CreateSessionFromArray(
            d->environment,
            std::data(onnx_data), std::size(onnx_data),
            session_options,
            &resource.session
        ));

        ortapi->ReleaseSessionOptions(session_options);

        if (auto err = checkSession(resource.session); err.has_value()) {
            return set_error(err.value());
        }

        auto input_shape = std::get<std::array<int64_t, 4>>(
            getShape(resource.session, true)
        );

#ifdef ENABLE_CUDA
        if (d->backend == Backend::CUDA) {
            resource.input.size = (
                input_shape[0] *
                input_shape[1] *
                input_shape[2] *
                input_shape[3]
            ) * getNumBytes(onnx_input_type);

            checkCUDAError(cudaMallocHost(
                &resource.input.h_data, resource.input.size,
                cudaHostAllocWriteCombined)
            );
            checkCUDAError(cudaMalloc(&resource.input.d_data, resource.input.size));

            checkError(ortapi->CreateTensorWithDataAsOrtValue(
                memory_info,
                resource.input.d_data, resource.input.size,
                std::data(input_shape), std::size(input_shape),
                static_cast<ONNXTensorElementDataType>(onnx_input_type),
                &resource.input_tensor
            ));
        } else
#endif // ENALBE_CUDA
        {
            checkError(ortapi->CreateTensorAsOrtValue(
                cpu_allocator,
                std::data(input_shape), std::size(input_shape),
                static_cast<ONNXTensorElementDataType>(onnx_input_type),
                &resource.input_tensor
            ));
        }

        auto output_shape = std::get<std::array<int64_t, 4>>(
            getShape(resource.session, false)
        );

#ifdef ENABLE_CUDA
        if (d->backend == Backend::CUDA) {
            resource.output.size = (
                output_shape[0] *
                output_shape[1] *
                output_shape[2] *
                output_shape[3]
            ) * getNumBytes(onnx_output_type);

            checkCUDAError(cudaMallocHost(&resource.output.h_data, resource.output.size));
            checkCUDAError(cudaMalloc(&resource.output.d_data, resource.output.size));

            checkError(ortapi->CreateTensorWithDataAsOrtValue(
                memory_info,
                resource.output.d_data, resource.output.size,
                std::data(output_shape), std::size(output_shape),
                static_cast<ONNXTensorElementDataType>(onnx_output_type),
                &resource.output_tensor
            ));
        } else
#endif // ENABLE_CUDA
        {
            checkError(ortapi->CreateTensorAsOrtValue(
                cpu_allocator,
                std::data(output_shape), std::size(output_shape),
                static_cast<ONNXTensorElementDataType>(onnx_output_type),
                &resource.output_tensor
            ));
        }

        checkError(ortapi->CreateIoBinding(resource.session, &resource.binding));

        checkError(ortapi->SessionGetInputName(
            resource.session, 0, cpu_allocator, &resource.input_name
        ));
        checkError(ortapi->SessionGetOutputName(
            resource.session, 0, cpu_allocator, &resource.output_name
        ));

        checkError(ortapi->BindInput(resource.binding, resource.input_name, resource.input_tensor));
        checkError(ortapi->BindOutput(resource.binding, resource.output_name, resource.output_tensor));

        if (auto err = checkNodesAndNetwork(resource.session, in_vis); err.has_value()) {
            return set_error(err.value());
        }

        if (i == 0) {
            setDimensions(d->out_vi, input_shape, output_shape, core, vsapi, onnx_output_type);
        }

        d->resources.push_back(resource);
    }

    ortapi->ReleaseMemoryInfo(memory_info);

    vsapi->createFilter(
        in, out, "Model",
        vsOrtInit, vsOrtGetFrame, vsOrtFree,
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
        "io.github.amusementclub.vs_onnxruntime", "ort",
        "ONNX Runtime ML Filter Runtime",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    registerFunc("Model",
        "clips:clip[];"
        "network_path:data;"
        "overlap:int[]:opt;"
        "tilesize:int[]:opt;"
        "provider:data:opt;" // "": Default (CPU), "CUDA": CUDA
        "device_id:int:opt;"
        "num_streams:int:opt;"
        "verbosity:int:opt;"
        "cudnn_benchmark:int:opt;"
        "builtin:int:opt;"
        "builtindir:data:opt;"
        "fp16:int:opt;"
        "path_is_serialization:int:opt;"
        "use_cuda_graph:int:opt;"
        "fp16_blacklist_ops:data[]:opt;"
        "prefer_nhwc:int:opt;"
        "output_format:int:opt;"
        "tf32:int:opt;"
        , vsOrtCreate,
        nullptr,
        plugin
    );

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);

        vsapi->propSetData(
            out, "onnxruntime_api_version_build",
            std::to_string(ORT_API_VERSION).c_str(), -1, paReplace
        );

        if (auto err = ortInit(); err.has_value()) {
            vsapi->logMessage(mtWarning, err.value().c_str());
        } else {
            if (auto p = OrtGetApiBase(); p) {
                vsapi->propSetData(
                    out, "onnxruntime_version",
                    p->GetVersionString(), -1, paReplace
                );
            }

            vsapi->propSetData(
                out, "onnxruntime_build_info",
                ortapi->GetBuildInfoString(), -1, paReplace
            );
        }

#ifdef ENABLE_CUDA
        vsapi->propSetData(
            out, "cuda_runtime_version",
            std::to_string(__CUDART_API_VERSION).c_str(), -1, paReplace
        );
#endif // ENABLE_CUDA

        vsapi->propSetData(
            out, "onnx_version",
            ONNX_NAMESPACE::LAST_RELEASE_VERSION, -1, paReplace
        );

        vsapi->propSetData(out, "path", vsapi->getPluginPath(myself), -1, paReplace);

#ifdef ENABLE_CUDA
        vsapi->propSetData(out, "providers", "CUDA", -1, paAppend);
#endif
#ifdef ENABLE_COREML
        vsapi->propSetData(out, "providers", "COREML", -1, paAppend);
#endif
#ifdef ENABLE_DML
        vsapi->propSetData(out, "providers", "DML", -1, paAppend);
#endif
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);
}
