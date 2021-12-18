#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <onnx/common/version.h>
#include <onnx/proto_utils.h>
#include <onnx/shape_inference/implementation.h>
#include <onnxruntime_c_api.h>

#include "config.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>

#define checkCUDAError(expr) do {                                              \
    if (cudaError_t result = expr; result != cudaSuccess) {                    \
        const char * error_str = cudaGetErrorString(result);                   \
        return set_error("'"s + # expr + "' failed: " + error_str);            \
    }                                                                          \
} while(0)
#endif // ENABLE_CUDA

#ifdef ENABLE_COREML
extern "C" OrtStatusPtr OrtSessionOptionsAppendExecutionProvider_CoreML(OrtSessionOptions *so, int flags);
#endif // ENABLE_COREML

#include <VapourSynth.h>
#include <VSHelper.h>

#ifdef _WIN32
#include <locale>
#include <codecvt>
static inline std::wstring translateName(const char *name) noexcept {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(name);
}
#else
#define translateName(n) (n)
#endif // _WIN32

#if not __cpp_lib_atomic_wait
#include <chrono>
#include <thread>
using namespace std::chrono_literals;
#endif

#define checkError(expr) do {                                                  \
    OrtStatusPtr __err = expr;                                                 \
    if (__err) {                                                               \
        const std::string message = ortapi->GetErrorMessage(__err);            \
        ortapi->ReleaseStatus(__err);                                          \
        return set_error("'"s + # expr + "' failed: " + message);              \
    }                                                                          \
} while(0)

using namespace std::string_literals;

static const VSPlugin * myself = nullptr;
static const OrtApi * ortapi = nullptr;

[[nodiscard]]
static std::optional<std::string> ortInit() noexcept {
    static std::once_flag ort_init_flag;

    std::call_once(ort_init_flag, []() {
        ortapi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
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
    checkError(ortapi->GetDimensions(tensor_info, shape.data(), std::size(shape)));

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


[[nodiscard]]
static std::optional<std::string> specifyShape(
    ONNX_NAMESPACE::ModelProto & model,
    int64_t tile_w,
    int64_t tile_h,
    int64_t batch = 1
) noexcept {

    ONNX_NAMESPACE::TensorShapeProto * input_shape {
        model
            .mutable_graph()
            ->mutable_input(0)
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
    };
    ONNX_NAMESPACE::TensorShapeProto * output_shape {
        model
            .mutable_graph()
            ->mutable_output(0)
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
    };

    constexpr auto n_idx = 0;
    constexpr auto h_idx = 2;
    constexpr auto w_idx = 3;

    input_shape->mutable_dim(n_idx)->set_dim_value(batch);
    input_shape->mutable_dim(h_idx)->set_dim_value(tile_h);
    input_shape->mutable_dim(w_idx)->set_dim_value(tile_w);

    output_shape->mutable_dim(n_idx)->set_dim_value(batch);
    output_shape->mutable_dim(h_idx)->clear_dim_value();
    output_shape->mutable_dim(w_idx)->clear_dim_value();

    // remove shape info
    model.mutable_graph()->mutable_value_info()->Clear();

    try {
        ONNX_NAMESPACE::shape_inference::InferShapes(model);
    } catch (const ONNX_NAMESPACE::InferenceError & e) {
        return e.what();
    }

    return {};
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
        if (vi->format->sampleType != stFloat || vi->format->bitsPerSample != 32) {
            return "expects clip with type fp32";
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

    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return set_error("expects network IO with type fp32");
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
    const VSAPI * vsapi
) noexcept {

    vi->height *= output_shape[2] / input_shape[2];
    vi->width *= output_shape[3] / input_shape[3];

    if (output_shape[1] == 1) {
        vi->format = vsapi->registerFormat(cmGray, stFloat, 32, 0, 0, core);
    } else if (output_shape[1] == 3) {
        vi->format = vsapi->registerFormat(cmRGB, stFloat, 32, 0, 0, core);
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
    CUDA = 1
};

#ifdef ENABLE_CUDA
struct CUDA_Resource_t {
    uint8_t * h_data;
    uint8_t * d_data;
    size_t size;
};
#endif // ENABLE_CUDA

struct Resource {
    OrtSession * session;
    OrtValue * input_tensor;
    OrtValue * output_tensor;
    OrtIoBinding * binding;

#ifdef ENABLE_CUDA
    cudaStream_t stream;
    CUDA_Resource_t input;
    CUDA_Resource_t output;
#endif // ENABLE_CUDA
};

struct vsOrtData {
    std::vector<VSNodeRef *> nodes;
    std::unique_ptr<VSVideoInfo> out_vi;

    int overlap_w, overlap_h;

    OrtEnv * environment;
    Backend backend;

#ifdef ENABLE_CUDA
    int device_id;
#endif // ENABLE_CUDA

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
                    checkCUDAError(cudaSetDevice(d->device_id));
                    checkCUDAError(cudaMemcpyAsync(
                        resource.input.d_data,
                        resource.input.h_data,
                        resource.input.size,
                        cudaMemcpyHostToDevice,
                        resource.stream
                    ));
                }
#endif // ENABLE_CUDA

                checkError(ortapi->RunWithBinding(resource.session, nullptr, resource.binding));

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

        for (const auto & frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        d->release(ticket);

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

    int device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
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

    const char * _provider = vsapi->propGetData(in, "provider", 0, &error);
    if (error) {
        _provider = "";
    }
    const std::string provider { _provider };

    int num_streams = int64ToIntS(vsapi->propGetInt(in, "num_streams", 0, &error));
    if (error) {
        num_streams = 1;
    }
    if (num_streams <= 0) {
        return set_error("\"num_streams\" must be positive");
    }

#if ENABLE_CUDA
    bool cudnn_benchmark = !!(vsapi->propGetInt(in, "cudnn_benchmark", 0, &error));
    if (error) {
        cudnn_benchmark = true;
    }
#endif

    if (auto err = ortInit(); err.has_value()) {
        return set_error(err.value());
    }

    checkError(ortapi->CreateEnv(verbosity, "vs-ort", &d->environment));

    std::string path { vsapi->propGetData(in, "network_path", 0, nullptr) };
    bool builtin = !!vsapi->propGetInt(in, "builtin", 0, &error);
    if (builtin) {
        const char *modeldir = vsapi->propGetData(in, "builtindir", 0, &error);
        if (!modeldir) modeldir = "models";
        path = std::string(modeldir) + "/" + path;
        std::string dir { vsapi->getPluginPath(myself) };
        dir = dir.substr(0, dir.rfind('/') + 1);
        path = dir + path;
    }

    std::string onnx_data;
    {
        std::ifstream onnx_stream(
            translateName(path.c_str()),
            std::ios::binary | std::ios::ate
        );

        if (!onnx_stream.good()) {
            return set_error("open "s + path + " failed"s);
        }

        onnx_data.resize(onnx_stream.tellg());
        onnx_stream.seekg(0, std::ios::beg);
        onnx_stream.read(onnx_data.data(), onnx_data.size());
    }

    ONNX_NAMESPACE::ModelProto onnx_proto;
    try {
        ONNX_NAMESPACE::ParseProtoFromBytes(
            &onnx_proto,
            onnx_data.data(), std::size(onnx_data)
        );
    } catch (const std::runtime_error & e) {
        return set_error(e.what());
    }

    if (auto err = specifyShape(onnx_proto, tile_w, tile_h); err.has_value()) {
        return set_error(err.value());
    }

    onnx_data = onnx_proto.SerializeAsString();
    if (std::size(onnx_data) == 0) {
        return set_error("proto serialization failed");
    }


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
        checkError(ortapi->EnableMemPattern(session_options));

        // TODO: other providers
        if (provider == "CPU") {
            ; // nothing to do
        }
#ifdef ENABLE_CUDA
        else if (provider == "CUDA") {
            d->backend = Backend::CUDA;
            checkError(ortapi->DisableMemPattern(session_options));
#ifdef _MSC_VER
            // Preload cuda dll from vsort directory.
            static std::once_flag cuda_dll_preloaded_flag;
            std::call_once(cuda_dll_preloaded_flag, []() {
                    extern void preloadCudaDlls();
                    preloadCudaDlls();
            });
#endif // _MSC_VER
            checkCUDAError(cudaSetDevice(device_id));
            d->device_id = device_id;

            checkCUDAError(cudaStreamCreate(&resource.stream));

            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = device_id;

#if ORT_API_VERSION >= 10
            if (cudnn_benchmark) {
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            } else {
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
            }
#else
            if (cudnn_benchmark) {
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::Exhaustive;
            } else {
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::Heuristic;
            }
#endif // ORT_API_VERSION >= 10

            cuda_options.user_compute_stream = resource.stream;
            cuda_options.has_user_compute_stream = 1;

            checkError(ortapi->SessionOptionsAppendExecutionProvider_CUDA(
                session_options,
                &cuda_options
            ));
        }
#endif // ENABLE_CUDA
#ifdef ENABLE_COREML
        else if (provider == "COREML") {
            checkError(OrtSessionOptionsAppendExecutionProvider_CoreML(
                session_options,
                0
            ));
        }
#endif // ENABLE_COREML
        else if (provider != "") {
            return set_error("unknwon provider " + provider);
        }

        checkError(ortapi->CreateSessionFromArray(
            d->environment,
            onnx_data.data(), std::size(onnx_data),
            session_options,
            &resource.session
        ));

        ortapi->ReleaseSessionOptions(session_options);

        if (auto err = checkSession(resource.session); err.has_value()) {
            return set_error(err.value());
        }

        OrtAllocator * allocator;
        checkError(ortapi->GetAllocatorWithDefaultOptions(&allocator));

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
            ) * sizeof(float);

            checkCUDAError(cudaMallocHost(
                &resource.input.h_data, resource.input.size,
                cudaHostAllocWriteCombined)
            );
            checkCUDAError(cudaMalloc(&resource.input.d_data, resource.input.size));

            OrtMemoryInfo * memory_info;
            checkError(ortapi->CreateMemoryInfo(
                "Cuda", OrtDeviceAllocator, device_id,
                OrtMemTypeDefault, &memory_info
            ));

            checkError(ortapi->CreateTensorWithDataAsOrtValue(
                memory_info,
                resource.input.d_data, resource.input.size,
                input_shape.data(), std::size(input_shape),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &resource.input_tensor
            ));

            ortapi->ReleaseMemoryInfo(memory_info);
        } else
#endif // ENALBE_CUDA
        {
            checkError(ortapi->CreateTensorAsOrtValue(
                allocator,
                input_shape.data(),
                std::size(input_shape),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
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
            ) * sizeof(float);

            checkCUDAError(cudaMallocHost(&resource.output.h_data, resource.output.size));
            checkCUDAError(cudaMalloc(&resource.output.d_data, resource.output.size));

            OrtMemoryInfo * memory_info;
            checkError(ortapi->CreateMemoryInfo(
                "Cuda", OrtDeviceAllocator, device_id,
                OrtMemTypeDefault, &memory_info
            ));

            checkError(ortapi->CreateTensorWithDataAsOrtValue(
                memory_info,
                resource.output.d_data, resource.output.size,
                output_shape.data(), std::size(output_shape),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &resource.output_tensor
            ));

            ortapi->ReleaseMemoryInfo(memory_info);
        } else
#endif // ENABLE_CUDA
        {
            checkError(ortapi->CreateTensorAsOrtValue(
                allocator,
                output_shape.data(),
                std::size(output_shape),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &resource.output_tensor
            ));
        }

        checkError(ortapi->CreateIoBinding(resource.session, &resource.binding));

        char * input_name;
        checkError(ortapi->SessionGetInputName(
            resource.session, 0, allocator, &input_name
        ));

        char * output_name;
        checkError(ortapi->SessionGetOutputName(
            resource.session, 0, allocator, &output_name
        ));

        checkError(ortapi->BindInput(resource.binding, input_name, resource.input_tensor));
        checkError(ortapi->BindOutput(resource.binding, output_name, resource.output_tensor));

        if (auto err = checkNodesAndNetwork(resource.session, in_vis); err.has_value()) {
            return set_error(err.value());
        }

        if (i == 0) {
            setDimensions(d->out_vi, input_shape, output_shape, core, vsapi);
        }

        d->resources.push_back(resource);
    }

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
        , vsOrtCreate,
        nullptr,
        plugin
    );

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);

        vsapi->propSetData(
            out, "onnxruntime_version",
            std::to_string(ORT_API_VERSION).c_str(), -1, paReplace
        );

        vsapi->propSetData(
            out, "cuda_runtime_version",
            std::to_string(__CUDART_API_VERSION).c_str(), -1, paReplace
        );

        vsapi->propSetData(
            out, "onnx_version",
            ONNX_NAMESPACE::LAST_RELEASE_VERSION, -1, paReplace
        );

        vsapi->propSetData(out, "path", vsapi->getPluginPath(myself), -1, paReplace);
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);
}
